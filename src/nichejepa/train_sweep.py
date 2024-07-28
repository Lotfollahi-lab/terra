"""
Adapted from Assran, M. et al. Self-supervised learning from images with a Joint-Embedding Predictive Architecture.
Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/train.py (05.06.2024).
"""

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass
import random
import copy
import logging
import sys
import yaml
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel

from .masks.multigene import MaskCollator
from .masks.utils import apply_masks
from .utils.distributed import (init_distributed,
                                AllReduce)
from .utils.logging import (CSVLogger,
                            gpu_timer,
                            grad_logger,
                            AverageMeter)
from .utils.tensors import repeat_interleave_batch
from .datasets.cell_neighborhood_dataset import make_cell_neighborhood_dataset 
from .helper import (load_checkpoint,
                     init_model,
                     init_opt)
from tqdm import tqdm
import pandas as pd
from .logistic_reg import logistic_
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from src.nichejepa.logistic_reg import logistic_
from src.nichejepa.nmi_ari import compute_nmi_ari

# Initialize shared list for collecting data
#manager = mp.Manager()
#data = manager.list()

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False,data=None):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    config = wandb.config
    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    pred_depth = int(config.pred_enc_depth %  10)
    pred_emb_dim = config.enc_emb_dim
    enc_depth = int( config.pred_enc_depth // 10)
    enc_emb_dim= config.enc_emb_dim
    top_layer=config.top_layer
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    batch_size = args['data']['batch_size']
    weighted_average = args['data']['weighted_average']
    if config.pred_enc_depth < 41:
        batch_size=130 
    elif config.pred_enc_depth <51:
        batch_size=100
    else:
        batch_size=30
    if config.epochs==0:
        batch_size=1000
    seq_len = args['data']['seq_len']
    seq_len_cell = args['data']['seq_len_cell']
    seq_len_neighborhood = args['data']['seq_len_neighborhood']
    just_cell = args['data']['just_cell']
    just_neighborhood = args['data']['just_neighborhood']
    has_cls = args['data']['has_cls']
    get_topk = args['data']['get_topk']
    top_k = 0
    if get_topk:
       top_k = config.top_k

    if just_cell and just_neighborhood:
         seq_len = seq_len_neighborhood + seq_len_cell
         if args['data']['has_cls']:
             seq_len+=2
    elif just_cell:
        seq_len = seq_len_cell
        if args['data']['has_cls']:
            seq_len+=1
    elif just_neighborhood:
        seq_len = seq_len_neighborhood
        if args['data']['has_cls']:
            seq_len+=1
    else:
        assert "both seq_len_niche and seq_len_cell can't be zero"
    vocab_size = args['data']['vocab_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    # --

    # -- MASK
    n_targets = config.n_targets
    n_contexts = args['mask']['n_contexts']
    target_mask_size = args['mask']['target_mask_size']
    context_mask_size = config.context_mask_size
    top_niche = args['mask']['top_niche']
    top_cell_type = args['mask']['top_cell_type']

    # --

    # -- OPTIMIZATION
    ema =[0,1]
    ema[0] = config.ema
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = config.epochs
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    seed = args['seed']
    folder = args['logging']['folder']+str(seed)+'/'
    os.makedirs(folder, exist_ok=True)
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #
    
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass
    
    # Initialize torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # Initialize encoder, predictor and target encoder
    encoder, predictor = init_model(
        device=device,
        seq_len=seq_len,
        enc_emb_dim=enc_emb_dim,
        enc_depth=enc_depth,
        vocab_size =vocab_size,
        pred_depth=pred_depth,
        pos_learnable=config.learnable,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        has_cls=args['data']['has_cls'])
    target_encoder = copy.deepcopy(encoder)

    # Initialize mask collator
    mask_collator = MaskCollator(seq_len=seq_len,
                                 target_mask_size = target_mask_size,
                                 context_mask_size = context_mask_size,
                                 n_targets=n_targets,
                                 n_contexts=n_contexts,
                                 has_cls=args['data']['has_cls'])
    
    # Initialize dataloader and -sampler
    data_path=args['data']['data_path']
    sample_size = args['data']['sample_size']
    dataset = load_from_disk(args['data']['data_path'])
    total_size = len(dataset)
    if sample_size == '100k':
            sample_size = min(100000, total_size)
    elif sample_size == '300k':
            sample_size = min(300000, total_size)
    else:
            sample_size = total_size
    random_state = 1
    rng = random.Random(random_state)
    sampled_indices = rng.sample(range(total_size), sample_size)
    dataset = dataset.select(sampled_indices)
    labels = dataset['cell_types']
    #train_indices, test_indices = train_test_split(range(len(dataset)), 
    #                                               test_size=args['data']['split'], 
    #                                               stratify=labels,
    #                                               random_state=1)
    train_indices, test_indices = train_test_split(range(len(dataset)),
                                                   test_size=args['data']['split'],
                                                   random_state=1)

    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)
    #dataset = dataset.train_test_split(test_size=args['data']['split'],
    #                                   seed=0)
    
    _, train_loader, test__sampler = make_cell_neighborhood_dataset(
            batch_size=batch_size,
            data=train_dataset,
            vocab_size=vocab_size,
            seq_len=seq_len,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            drop_last=False,
            just_cell=just_cell,
            just_neighborhood=just_neighborhood,
            seq_len_cell = seq_len_cell,
            seq_len_neighborhood = seq_len_neighborhood,
            has_cls =args['data']['has_cls'])

    _, test_loader, train__sampler = make_cell_neighborhood_dataset(
            batch_size=batch_size,
            data=test_dataset,
            vocab_size=vocab_size,
            seq_len=seq_len,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=False,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            drop_last=False,
            just_cell=just_cell,
            just_neighborhood=just_neighborhood,
            seq_len_cell = seq_len_cell,
            seq_len_neighborhood = seq_len_neighborhood,
            has_cls =args['data']['has_cls'])

    ipe = len(train_loader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # Load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # Initialize wandb project

    # Run training loop
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}")

        # Update distributed dataloader epoch
        train__sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(train_loader):
            def load_cell_neighborhoods():
                # -- unsupervised imgs
                cell_neighborhood_tokens = udata[0].to(device, non_blocking=True)
                seg_label = udata[1].to(device, non_blocking=True)                 
                niche_label = udata[2]
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (cell_neighborhood_tokens, seg_label, niche_label,  masks_1, masks_2)
            cell_neighborhood_tokens, seg_label, niche_label, masks_enc, masks_pred = load_cell_neighborhoods()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target():
                    with torch.no_grad(): # no backward pass for target encoder
                        # Encode all cell neighborhood tokens
                        h = target_encoder(cell_neighborhood_tokens, seg_label) # output (B, seq_len, emb_size)
                        # Normalize over feature dim
                        h = F.layer_norm(h, (h.size(-1),)) # output (B, seq_len, emb_size)
                        # Only keep encoded targets (masked genes of h)
                        h = apply_masks(h, masks_pred) # output (B * n_targets, target_size, emb_size)
                        B = len(h)
                        # Repeat targets if multiple contexts
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc)) # output (B * n_targets * n_contexts, target_size, emb_size)
                        return h

                def forward_context():
                    # Encode only context cell neighborhood tokens
                    z = encoder(cell_neighborhood_tokens, seg_label, masks_enc) # output (B, min_context_size, emb_size) where min_context size is minmum context size in the batch after removal of overlapping targets
                    z = predictor(z, masks_enc, masks_pred) # output (B * n_targets, target_size, emb_size)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1: forward pass
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)

                # Step 2: backward pass and step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3: momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))
            log_stats()
            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)
    if data==None:
        data=[]
    encoder.eval()
    target_encoder.eval()
    def process_loader(loader, dataset_type,top_k=0):
        for itr, (udata, masks_enc, masks_pred) in tqdm(enumerate(loader)):
                def load_cell_neighborhoods():
                    # -- unsupervised loader
                    cell_neighborhood_tokens = udata[0].to(device, non_blocking=True)
                    seg_label = udata[1].to(device, non_blocking=True)
                    if len(udata)==4:
                       niche_label = udata[2]
                       cell_type = udata[3]  # Assuming udata[3] is cell_type
                    elif len(udata)==3:
                       if just_cell:
                          niche_label = None
                          cell_type = udata[2]                       
                       else:
                           cell_type = None
                           niche_label = udata[2]
                    masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                    masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                    return (cell_neighborhood_tokens, seg_label, niche_label, cell_type, masks_1, masks_2)
                cell_neighborhood_tokens, seg_label, niche_label, cell_type, masks_enc, masks_pred = load_cell_neighborhoods()

                def eval_step():
                    def forward_context(top_index, label_name, label_value):
                        # Encode all cell neighborhood tokens
                        if num_epochs == 0:
                           z_list = [target_encoder(cell_neighborhood_tokens,seg_label,just_emb=True,multi_layer=True)]
                        else:
                           z_list = target_encoder(cell_neighborhood_tokens, seg_label,multi_layer=True)
                        average_features_list = []
                        for z in z_list[top_layer-1:]:
                          masks = (cell_neighborhood_tokens != 0).int()
                          if just_cell and just_neighborhood:
                            if label_name == "niche_type":
                                masks[:, 0:seq_len_cell] = 0
                                if get_topk:
                                   masks[:, seq_len_cell+top_k:] = 0
                            elif label_name == "cell_type":
                                masks[:, seq_len_cell:] = 0
                          if has_cls:
                            masks[:, :] = 0
                            if label_name == "cell_type":
                                masks[:, 0] = 1
                            if label_name == "niche_type":
                                masks[:, -1] = 1
                          if get_topk:
                              masks[:, top_k:] = 0
                          expanded_mask = masks.unsqueeze(-1).expand_as(z)
                          if weighted_average:
                             rank = torch.zeros_like(cell_neighborhood_tokens, dtype=torch.float)
                             for i in range(cell_neighborhood_tokens.size(0)):
                                 non_zero_indices = torch.nonzero(cell_neighborhood_tokens[i, :] != 0, as_tuple=True)[0]
                                 rank[i, non_zero_indices] = torch.arange(1, len(non_zero_indices) + 1, dtype=torch.float, device=cell_neighborhood_tokens.device)
                             rank_max = rank.max(dim=1, keepdim=True)[0]
                             rank_sum = rank.sum(dim=1, keepdim=True)
                             weights = (rank_max - rank + 1) / rank_sum
                             weights = weights.unsqueeze(-1).expand_as(z)
                             expanded_mask = expanded_mask*weights
                          masked_features = z * expanded_mask
                          summed_features = masked_features.sum(dim=1)
                          if weighted_average:
                             average_features = summed_features
                          else:
                             count_valid_positions = expanded_mask.sum(dim=1)
                             average_features = summed_features / count_valid_positions.clamp(min=1)
                             average_features[count_valid_positions == 0] = 0
                          average_features_list.append(average_features.cpu().numpy())
                        average_features = np.concatenate(average_features_list, axis=1)
                        label_cpu = label_value
                        for i in range(len(average_features)):
                            sample_features = average_features[i]
                            sample_label = label_cpu[i]
                            data_dict = {
                                'split': dataset_type,
                                'label_name': label_name,
                                'seed': seed,
                                label_name: sample_label
                            }
                            for j, feature in enumerate(sample_features):
                                data_dict[f'feature_{j}'] = feature
                            data.append(data_dict)

                    with torch.no_grad():
                        if just_neighborhood:
                           forward_context(seq_len, "niche_type", niche_label)
                        if just_cell:
                           forward_context(seq_len_cell, "cell_type", cell_type)

                eval_step()
    process_loader(train_loader, 'train')
    process_loader(test_loader, 'test')
    ''' 
    results_for_different_k = []
    nmi_for_different_k = []
    ari_for_different_k = []
    #786
    for k in tqdm(range(1,786)):
       process_loader(train_loader, 'train',top_k=k)
       process_loader(test_loader, 'test',top_k=k)
       final_df = pd.DataFrame(list(data))
       print(final_df)
       nmi_ari_out = compute_nmi_ari(final_df,config.enc_emb_dim,'niche_type')
       test_f1_cell, test_f1_niche = logistic_(final_df,num_features=config.enc_emb_dim)
       #print(test_f1_niche)
       results_for_different_k.append(test_f1_niche)
       print(nmi_ari_out)
       nmi_for_different_k.append(nmi_ari_out.loc[0,'nmi_score'])
       ari_for_different_k.append(nmi_ari_out.loc[0,'ari_score'])
       print(len(data))
       data=[]
    #print(results_for_different_k)
    with open('results_for_different_k_merfish.pkl', 'wb') as file:
           pickle.dump(results_for_different_k, file)
    print(nmi_for_different_k)
    print(ari_for_different_k)
    with open('nmi_for_different_k_merfish.pkl', 'wb') as file:
            pickle.dump(nmi_for_different_k, file)
    with open('ari_for_different_k_merfish.pkl', 'wb') as file:
            pickle.dump(ari_for_different_k, file)
    '''
if __name__ == "__main__":
    main()
