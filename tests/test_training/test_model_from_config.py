"""
Tests for rebuilding a trained model from its saved config.

Finetuning rebuilt the encoder with ``init_model``'s own defaults instead of
the training config: a TERRA-96M-like model (no ``meta.mlp_bias`` key,
trained with the ``train.py`` default ``True``) silently lost its value-MLP
biases, and a TERRA-112M-like model (``data.n_special_values``,
``data.nz_spc``, ``meta.new_spc``) could not be loaded. Every entry point
that loads a trained checkpoint now builds the model with
``init_model_from_config``.

Finetuning also hard-coded the seed 0; the optional top-level ``seed`` config
key now sets it, as for pretraining. Supervised finetuning saw the training
cells in the same order in every epoch; it now shuffles them in every epoch
with that seed (``finetune.shuffle: false`` keeps the order).
"""

import copy
import pickle

import anndata as ad
import numpy as np
import pytest
import torch
import yaml

from terra.utils.helper import init_model, init_model_from_config

N_GENES = 60
SPV_KEYS = ['spv_ds1_b0', 'spv_ds1_b1', 'spv_xenium', 'spv_human', 'spv_lung']

# Layouts of the released configs (TERRA-112M / TERRA-96M), scaled down
CONFIG_112M = {
    'data': {
        'dataset_name': 'toy', 'model_input_size': 64, 'n_neighs': 3,
        'rank_cell_norm_method': None, 'rank_gene_norm_method': None,
        'rank_count_norm_method': None, 'count_cell_norm_method': None,
        'count_gene_norm_method': None, 'count_count_norm_method': 'shifted_log',
        'tokenizer_type': 'cell_graph', 'seq_len_cell': 16,
        'seq_len_neighborhood': 48, 'n_segments': 4, 'sampling_strategy': None,
        'batch_size': 8, 'num_workers': 0, 'pin_memory': False,
        'sep_gene_tokens_neb': False, 'n_special_values': 32, 'nz_spc': True},
    'meta': {
        'api_version': 'v3', 'gt_type': 'combined', 'count_encoding': 'value_bins',
        'n_value_bins': 20, 'cell_pos_enc': 'segment', 'enc_depth': 1,
        'enc_emb_dim': 16, 'pred_depth': 1, 'pred_emb_dim': 8, 'num_heads': 2,
        'mlp_ratio': 4, 'special_tokens': ['batch'], 'pos_learnable': False,
        'use_flash_attention': True, 'use_layer_norm': True, 'predict_gene': True,
        'new_spc': True, 'restrict_special_attention': False},
    'mask': {'n_contexts': 1, 'n_targets': 1, 'per_block_mask_ratio': 0.6},
}
# No metatoken, MLP count encoding; like the released config it has no
# meta.mlp_bias key (the model was trained with mlp_bias=True)
CONFIG_96M = copy.deepcopy(CONFIG_112M)
CONFIG_96M['data'].pop('n_special_values')
CONFIG_96M['data'].pop('nz_spc')
CONFIG_96M['meta'].update({'special_tokens': [], 'count_encoding': 'mlp',
                           'cell_pos_enc': 'coord'})
CONFIG_96M['meta'].pop('new_spc')

# How the models were built in training (train.py reads every key, with
# mlp_bias defaulting to True)
TRAINING_KWARGS = {
    '112M': dict(n_special_values=32, nz_spc=True, new_spc=True, mlp_bias=True),
    '96M': dict(n_special_values=0, nz_spc=False, new_spc=False, mlp_bias=True),
}
CONFIGS = {'112M': CONFIG_112M, '96M': CONFIG_96M}


def _token_dict():
    """Token dictionary with special tokens at the ids the datasets expect."""
    token_dict = {'<pad>': 0}
    for i in range(11):
        token_dict[f'cls_{i}'] = len(token_dict)
    while len(token_dict) < 103:
        token_dict[f'reserved_{len(token_dict)}'] = len(token_dict)
    for key in ['spt_tissue', 'spt_assay', 'spt_gene_panel', 'spt_batch']:
        token_dict[key] = len(token_dict)
    genes = [f'ENSG{i + 1:011d}' for i in range(N_GENES)]
    for gene in genes:
        token_dict[gene] = len(token_dict)
    # Deliberately fewer spv_* keys than the configured n_special_values
    for key in SPV_KEYS + [f'spv_gene_panel{N_GENES}']:
        token_dict[key] = len(token_dict)
    return token_dict, genes


def _training_model(config, layout, token_dict, **overrides):
    """Build encoder and predictor the way ``train.py`` built them."""
    meta, data = config['meta'], config['data']
    n_special_tokens = len(meta['special_tokens'])
    return init_model(
        gt_type=meta['gt_type'], count_encoding=meta['count_encoding'],
        n_value_bins=meta['n_value_bins'], cell_pos_enc=meta['cell_pos_enc'],
        device='cpu', vocab_size=len(token_dict),
        seq_len=(data['seq_len_cell'] + data['seq_len_neighborhood'] +
                 n_special_tokens),
        n_special_tokens=n_special_tokens, n_segments=data['n_segments'],
        enc_emb_dim=meta['enc_emb_dim'], enc_depth=meta['enc_depth'],
        pred_emb_dim=meta['pred_emb_dim'], pred_depth=meta['pred_depth'],
        num_heads=meta['num_heads'], mlp_ratio=meta['mlp_ratio'],
        use_flash_attention=meta['use_flash_attention'],
        use_layer_norm=meta['use_layer_norm'],
        sep_gene_tokens_neb=data['sep_gene_tokens_neb'],
        predict_gene=meta['predict_gene'], pos_learnable=meta['pos_learnable'],
        **{**TRAINING_KWARGS[layout], **overrides})


def _perturb(module, seed):
    """Make every parameter (incl. biases) distinguishable from its init."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.add_(0.1 * torch.randn(param.shape, generator=generator))


@pytest.mark.parametrize('layout', ['112M', '96M'])
def test_builder_loads_training_checkpoint_strictly(layout):
    token_dict, _ = _token_dict()
    torch.manual_seed(0)
    encoder, predictor = _training_model(CONFIGS[layout], layout, token_dict)
    _perturb(encoder, 1)
    _perturb(predictor, 2)

    torch.manual_seed(1)
    rebuilt_encoder, rebuilt_predictor = init_model_from_config(
        CONFIGS[layout], token_dict, 'cpu')
    # Strict loading: no missing, unexpected or mis-shaped weights
    rebuilt_encoder.load_state_dict(encoder.state_dict())
    rebuilt_predictor.load_state_dict(predictor.state_dict())
    if layout == '96M':
        assert 'backbone.value_embed.fc1.bias' in rebuilt_encoder.state_dict()
    else:
        assert rebuilt_predictor.backbone.new_spc


def test_builder_counts_special_values_for_older_configs():
    """Models trained before ``data.n_special_values`` existed sized their
    special value table by the ``spv`` entries of the token dictionary."""
    token_dict, _ = _token_dict()
    n_spv = sum(1 for key in token_dict if 'spv' in key)
    config = copy.deepcopy(CONFIG_112M)
    config['data'].pop('n_special_values')
    torch.manual_seed(0)
    encoder, predictor = _training_model(
        config, '112M', token_dict, n_special_values=n_spv)

    rebuilt_encoder, rebuilt_predictor = init_model_from_config(
        config, token_dict, 'cpu')
    rebuilt_encoder.load_state_dict(encoder.state_dict())
    rebuilt_predictor.load_state_dict(predictor.state_dict())


def test_builder_rejects_models_trained_with_cls_tokens():
    config = copy.deepcopy(CONFIG_112M)
    config['meta']['add_cls'] = True
    token_dict, _ = _token_dict()
    with pytest.raises(ValueError, match='add_cls'):
        init_model_from_config(config, token_dict, 'cpu')


@pytest.mark.parametrize('layout', ['112M', '96M'])
def test_builder_matches_previous_embedding_construction(layout):
    """The released models embed exactly as before (embed.py's old call)."""
    config = CONFIGS[layout]
    meta, data = config['meta'], config['data']
    token_dict, _ = _token_dict()
    n_special_tokens = len(meta['special_tokens'])
    torch.manual_seed(0)
    previous, _ = init_model(
        gt_type=meta['gt_type'], count_encoding=meta['count_encoding'],
        n_value_bins=meta['n_value_bins'], cell_pos_enc=meta['cell_pos_enc'],
        device='cpu', vocab_size=len(token_dict),
        seq_len=(data['seq_len_cell'] + data['seq_len_neighborhood'] +
                 n_special_tokens),
        n_special_tokens=n_special_tokens, n_segments=data['n_segments'],
        n_special_values=data.get('n_special_values', 0),
        enc_emb_dim=meta['enc_emb_dim'], enc_depth=meta['enc_depth'],
        pred_emb_dim=meta['pred_emb_dim'], pred_depth=meta['pred_depth'],
        num_heads=meta['num_heads'], mlp_ratio=meta['mlp_ratio'],
        use_flash_attention=meta['use_flash_attention'],
        api_version=meta['api_version'],
        sep_gene_tokens_neb=data['sep_gene_tokens_neb'],
        predict_gene=meta['predict_gene'], pos_learnable=meta['pos_learnable'],
        nz_spc=data.get('nz_spc', False), mlp_bias=meta.get('mlp_bias', True))
    torch.manual_seed(0)
    rebuilt, _ = init_model_from_config(config, token_dict, 'cpu')

    assert str(rebuilt) == str(previous)
    previous_state, rebuilt_state = previous.state_dict(), rebuilt.state_dict()
    assert list(rebuilt_state) == list(previous_state)
    for key, value in previous_state.items():
        assert torch.equal(rebuilt_state[key], value), key


# -- Finetuning end to end on tiny bundles ----------------------------------- #

def _write_bundle(folder, layout):
    folder.mkdir()
    config = copy.deepcopy(CONFIGS[layout])
    token_dict, genes = _token_dict()
    with open(folder / 'token_dictionary.pkl', 'wb') as f:
        pickle.dump(token_dict, f)
    with open(folder / 'ensembl_dictionary.pkl', 'wb') as f:
        pickle.dump({f'GENE{i}': gene for i, gene in enumerate(genes)}, f)
    with open(folder / 'gene_count_dictionary.pkl', 'wb') as f:
        pickle.dump({gene: 1000 for gene in genes}, f)
    config['data']['token_dict_folder_path'] = str(
        folder / 'token_dictionary.pkl')
    with open(folder / 'model_config.yaml', 'w') as f:
        yaml.safe_dump(config, f)
    torch.manual_seed(0)
    encoder, predictor = _training_model(CONFIGS[layout], layout, token_dict)
    _perturb(encoder, 1)
    _perturb(predictor, 2)
    # DDP-style keys, as saved in training
    encoder_state = {f'module.{k}': v for k, v in encoder.state_dict().items()}
    predictor_state = {
        f'module.{k}': v for k, v in predictor.state_dict().items()}
    torch.save({'encoder': encoder_state, 'predictor': predictor_state,
                'target_encoder': encoder_state, 'epoch': 0},
               folder / 'model_checkpoint.pt')
    return encoder


def _tokenized_data(bundle, cache, include_special_tokens=False):
    from terra import harmonize_adata, tokenize_adata

    rng = np.random.default_rng(0)
    n_cells = 48
    adata = ad.AnnData(
        X=rng.poisson(2.0, size=(n_cells, N_GENES)).astype(np.float32))
    adata.var_names = [f'GENE{i}' for i in range(N_GENES)]
    adata.obs_names = [f'c{i}' for i in range(n_cells)]
    adata.obs['sample'] = 'S1'
    adata.obs['label'] = rng.integers(0, 3, size=n_cells)
    adata.obsm['spatial'] = rng.uniform(
        0, 1000, size=(n_cells, 2)).astype(np.float32)
    adata.uns.update({'dataset_id': 'ds1', 'batch': 'b0', 'assay': 'xenium',
                      'species': 'human', 'tissue': 'lung'})
    harmonized = harmonize_adata(
        adata, gene_mapping_dict_file_path=str(bundle / 'ensembl_dictionary.pkl'),
        min_genes_per_cell=0, min_cells_per_gene=0)
    dataset = tokenize_adata(
        harmonized, model_folder_path=str(bundle), cache_directory_path=str(cache),
        nproc=1, processing_mode='sequential',
        include_special_tokens=include_special_tokens)
    return harmonized, dataset


def _split(harmonized, dataset):
    cell_ids = list(harmonized.obs['cell_id'])
    parts = {}
    for name, ids in {'train': cell_ids[:32], 'val': cell_ids[32:40],
                      'test': cell_ids[40:]}.items():
        ids = set(ids)
        parts[name] = (
            harmonized[harmonized.obs['cell_id'].isin(ids)].copy(),
            dataset.filter(lambda example: example['cell_id'] in ids))
    return parts


@pytest.mark.parametrize('layout, keep_special_tokens', [
    ('96M', False), ('112M', False), ('112M', True)])
def test_finetune_uses_pretrained_weights(tmp_path, monkeypatch, layout,
                                          keep_special_tokens):
    import terra.training.finetune as finetune_module

    bundle = tmp_path / 'bundle'
    pretrained = _write_bundle(bundle, layout)
    parts = _split(*_tokenized_data(
        bundle, tmp_path / 'cache', include_special_tokens=keep_special_tokens))

    captured = {}
    base_class = finetune_module.ClassificationModel

    class Spy(base_class):
        def __init__(self, base_model, **kwargs):
            captured['state'] = {
                k: v.detach().clone() for k, v in base_model.state_dict().items()}
            super().__init__(base_model=base_model, **kwargs)

    monkeypatch.setattr(finetune_module, 'ClassificationModel', Spy)
    args = {
        'model': {'pretrained_checkpoint_path': str(bundle)},
        'data': {'label_name': 'label', 'batch_size': 8, 'pin_memory': False,
                 'num_workers': 0},
        'finetune': {'use_peft': False, 'peft_target_modules': ['0.attn.qkv'],
                     'use_mlp': False, 'hidden_dim': 8, 'lr': 1e-4,
                     'num_epochs': 1, 'patience': 10, 'save_every_k_epochs': -1,
                     'selection_type': 'agg_graph', 'excluded_tokens': None,
                     'top_k': None, 'keep_special_tokens': keep_special_tokens}}
    predictions, targets, accuracy = finetune_module.finetune(
        args=args,
        train_adata=parts['train'][0], train_dataset=parts['train'][1],
        val_adata=parts['val'][0], val_dataset=parts['val'][1],
        test_adata=parts['test'][0], test_dataset=parts['test'][1])

    assert len(predictions) == len(targets) == 8
    assert 0.0 <= accuracy <= 1.0
    expected = pretrained.state_dict()
    assert sorted(captured['state']) == sorted(expected)
    for key, value in expected.items():
        assert torch.equal(captured['state'][key], value), key


@pytest.mark.parametrize('layout', ['112M', '96M'])
def test_finetune_self_supervised_loads_released_layouts(tmp_path, layout):
    from terra.training.finetune_self_supervised import finetune_self_supervised

    bundle = tmp_path / 'bundle'
    _write_bundle(bundle, layout)
    _, dataset = _tokenized_data(bundle, tmp_path / 'cache')
    args = {
        'model': {'pretrained_checkpoint_path': str(bundle)},
        'data': {'batch_size': 8, 'pin_memory': False, 'num_workers': 0,
                 'sample_gene_masks': True, 'sample_segments': False},
        'finetune': {'use_peft': True, 'peft_rank': 2, 'peft_alpha': 4,
                     'peft_dropout': 0.0, 'peft_bias': 'none', 'lr': 1e-4,
                     'num_epochs': 1, 'use_bfloat16': False}}
    run_dir = finetune_self_supervised(
        args=args, dataset=dataset, save_folder_path=str(tmp_path / 'runs'))
    assert any((tmp_path / 'runs').iterdir())
    assert run_dir


def test_finetune_self_supervised_samples_gene_masks_by_default(tmp_path):
    # Without data.sample_gene_masks the collator returned no masks and the
    # first step failed with a TypeError
    from terra.training.finetune_self_supervised import finetune_self_supervised

    bundle = tmp_path / 'bundle'
    _write_bundle(bundle, '96M')
    _, dataset = _tokenized_data(bundle, tmp_path / 'cache')
    args = {
        'model': {'pretrained_checkpoint_path': str(bundle)},
        'data': {'batch_size': 8, 'pin_memory': False, 'num_workers': 0},
        'finetune': {'use_peft': False, 'lr': 1e-4, 'num_epochs': 1,
                     'use_bfloat16': False}}
    run_dir = finetune_self_supervised(
        args=args, dataset=dataset, save_folder_path=str(tmp_path / 'runs'))
    assert list((tmp_path / 'runs').rglob('checkpoint_epoch_1.pt'))
    assert run_dir


def test_selection_mask_never_selects_special_tokens():
    from terra.training.finetune import build_selection_mask

    config = CONFIG_112M
    n_tokens = (config['data']['seq_len_cell'] +
                config['data']['seq_len_neighborhood'])
    tokens = torch.randint(110, 150, (2, 1 + n_tokens))
    mask = build_selection_mask(tokens, 1, config, 'agg_graph', None, None)
    assert mask.shape == tokens.shape
    assert not mask[:, 0].any()
    assert mask[:, 1:].any()


def _finetune_args(bundle, **extra):
    args = {
        'model': {'pretrained_checkpoint_path': str(bundle)},
        'data': {'label_name': 'label', 'batch_size': 8, 'pin_memory': False,
                 'num_workers': 0},
        'finetune': {'use_peft': False, 'peft_target_modules': ['0.attn.qkv'],
                     'use_mlp': False, 'hidden_dim': 8, 'lr': 1e-4,
                     'num_epochs': 1, 'patience': 10, 'save_every_k_epochs': -1,
                     'selection_type': 'agg_graph', 'excluded_tokens': None,
                     'top_k': None}}
    args.update(extra)
    return args


def test_finetune_seed_is_configurable(tmp_path, monkeypatch):
    import terra.training.finetune as finetune_module

    bundle = tmp_path / 'bundle'
    _write_bundle(bundle, '96M')
    parts = _split(*_tokenized_data(bundle, tmp_path / 'cache'))

    heads = []
    base_class = finetune_module.ClassificationModel

    class Spy(base_class):
        def __init__(self, base_model, **kwargs):
            super().__init__(base_model=base_model, **kwargs)
            heads.append(self.classification_head.weight.detach().clone())

    monkeypatch.setattr(finetune_module, 'ClassificationModel', Spy)
    for extra in ({}, {'seed': 0}, {'seed': 1}):
        finetune_module.finetune(
            args=_finetune_args(bundle, **extra),
            train_adata=parts['train'][0], train_dataset=parts['train'][1],
            test_adata=parts['test'][0], test_dataset=parts['test'][1])

    # Without the key the historical seed 0 is used
    assert torch.equal(heads[0], heads[1])
    assert not torch.equal(heads[1], heads[2])


def _training_orders(monkeypatch, finetune_module, parts, args):
    """Run supervised finetuning and return the cell order of every epoch."""
    orders, loaders = [], []
    make_loader = finetune_module.init_dataloader_and_sampler

    class RecordingLoader:
        def __init__(self, loader):
            self.loader = loader

        def __len__(self):
            return len(self.loader)

        def __iter__(self):
            orders.append([])
            for item in self.loader:
                orders[-1].extend(item[0]['cell_id'])
                yield item

    def make_recording_loader(*args, **kwargs):
        loader, sampler = make_loader(*args, **kwargs)
        loaders.append(loader)
        # The training loader is created first
        if len(loaders) == 1:
            loader = RecordingLoader(loader)
        return loader, sampler

    monkeypatch.setattr(finetune_module, 'init_dataloader_and_sampler',
                        make_recording_loader)
    finetune_module.finetune(
        args=args, train_adata=parts['train'][0],
        train_dataset=parts['train'][1], test_adata=parts['test'][0],
        test_dataset=parts['test'][1])
    return orders


def test_finetune_shuffles_the_training_data_every_epoch(tmp_path, monkeypatch):
    import terra.training.finetune as finetune_module

    bundle = tmp_path / 'bundle'
    _write_bundle(bundle, '96M')
    parts = _split(*_tokenized_data(bundle, tmp_path / 'cache'))
    in_order = list(parts['train'][1]['cell_id'])

    def orders(seed=0, **finetune_args):
        args = _finetune_args(bundle, seed=seed)
        args['finetune'].update(num_epochs=2, **finetune_args)
        return _training_orders(monkeypatch, finetune_module, parts, args)

    shuffled = orders()
    assert len(shuffled) == 2
    for order in shuffled:
        assert sorted(order) == sorted(in_order)
        assert order != in_order
    assert shuffled[0] != shuffled[1]
    # Reproducible with the seed, different with another one
    assert orders() == shuffled
    assert orders(seed=1) != shuffled
    # finetune.shuffle False keeps the order of the data
    assert orders(shuffle=False) == [in_order, in_order]


def test_self_supervised_dataset_shuffle_follows_seed(tmp_path):
    from datasets import Dataset

    from terra.training.finetune_self_supervised import _load_finetune_dataset

    paths = []
    for name in ('section_a', 'section_b'):
        path = tmp_path / name
        Dataset.from_dict({'cell_id': [f'{name}_{i}' for i in range(20)],
                           'tokens': [[i] for i in range(20)]}).save_to_disk(path)
        paths.append(str(path))

    def order(**kwargs):
        return list(_load_finetune_dataset(paths, **kwargs)['cell_id'])

    assert order() == order(seed=0)
    assert order(seed=0) != order(seed=1)


def test_finetune_evaluates_without_autograd_and_bf16_is_opt_in(
        tmp_path, monkeypatch):
    """Validation and test passes run without autograd (memory); the model
    runs under bf16 autocast only with finetune.use_bfloat16."""
    import terra.training.finetune as finetune_module

    bundle = tmp_path / 'bundle'
    _write_bundle(bundle, '96M')
    parts = _split(*_tokenized_data(bundle, tmp_path / 'cache'))

    grad_enabled_in_eval = []
    base_class = finetune_module.ClassificationModel

    class Spy(base_class):
        def forward(self, *args, **kwargs):
            if not self.training:
                grad_enabled_in_eval.append(torch.is_grad_enabled())
            return super().forward(*args, **kwargs)

    monkeypatch.setattr(finetune_module, 'ClassificationModel', Spy)
    requested = []
    autocast = torch.amp.autocast

    def recording_autocast(device_type, dtype=None, enabled=True, **kwargs):
        requested.append((device_type, dtype, enabled))
        # (CPU test: CUDA autocast would be a no-op anyway)
        return autocast(device_type, dtype=dtype, enabled=False, **kwargs)

    monkeypatch.setattr(torch.amp, 'autocast', recording_autocast)
    for use_bfloat16 in (None, True):
        requested.clear()
        args = _finetune_args(bundle)
        if use_bfloat16 is not None:
            args['finetune']['use_bfloat16'] = use_bfloat16
        finetune_module.finetune(
            args=args,
            train_adata=parts['train'][0], train_dataset=parts['train'][1],
            val_adata=parts['val'][0], val_dataset=parts['val'][1],
            test_adata=parts['test'][0], test_dataset=parts['test'][1])
        assert requested
        assert set(requested) == {('cuda', torch.bfloat16,
                                   bool(use_bfloat16))}
    assert grad_enabled_in_eval and not any(grad_enabled_in_eval)
