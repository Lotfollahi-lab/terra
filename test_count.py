import anndata as ad
from app.count_decoder import apply_count_decoder
from app.count_decoder import _spot_level_metrics, _apply_metric_transform, _to_dense
import numpy as np

adata_path = "/nfs/team361/sb75/DATASETS/gold/cell-graph-tokenizer/kidney_perturbation/nemokidneyxeniumatlas_annotated_processed.h5ad"
checkpoint_path = "results/zinb_decoder.pt"  # or a folder with zinb_decoder.pt inside
emb_key = "cell_emb"
checkpoint_path = "results_neb/zinb_decoder.pt"  # or a folder with zinb_decoder.pt inside
emb_key="neighborhood_emb"

adata = ad.read_h5ad(adata_path)

adata = apply_count_decoder(
    adata,
    emb_key=emb_key,
    model_folder_path=checkpoint_path,  # can be folder or exact file
    decoded_counts_layer_key="decoded_counts",
    # optional overrides:
    # loss_type="nb_softplus",
    # device=0,
    # batch_size=1024,
)
 
preds = _to_dense(adata.layers["decoded_counts"])
target = _to_dense(adata.layers["counts"])
target = _to_dense(adata.layers["counts_neighborhood"])

# optional: same transform used in training metrics
preds = _apply_metric_transform(preds, "none")  # or "log1p"
target = _apply_metric_transform(target, "none")

spot_metrics = _spot_level_metrics(preds, target)
print(spot_metrics)

adata.write_h5ad("/path/to/output_with_decoded_counts.h5ad")
