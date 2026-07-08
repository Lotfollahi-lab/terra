import argparse
import pickle
import sys
from pathlib import Path

from datasets import load_from_disk


ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from terra.inference import infer_token_distance  # noqa: E402


MODEL_FOLDER_PATH = (
    "/nfs/team361/sb75/nichejepa-reproducibility/artifacts/models/01112025_081731_614"
)


def load_dataset(dataset_path: str):
    dataset = load_from_disk(dataset_path)
    dataset.set_format(type="torch")
    return dataset


def save_result(result, output_path: str | None) -> None:
    if output_path is None:
        return
    with open(output_path, "wb") as handle:
        pickle.dump(result, handle)


def run_infer_token_distance(args: argparse.Namespace):
    dataset_original = load_dataset(args.dataset_original_path)
    dataset_perturbed = load_dataset(args.dataset_perturbed_path)
    return infer_token_distance(
        dataset_original=dataset_original,
        dataset_perturbed=dataset_perturbed,
        model_folder_path=args.model_folder_path,
        emb_layer=args.emb_layer,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        loss=args.loss,
        p=args.p,
        blur=args.blur,
        backend=args.backend,
        device=args.device,
        ignore_spc_tokens=not args.keep_special_tokens,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple score runner using terra.inference APIs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    token_distance = subparsers.add_parser("infer_token_distance")
    token_distance.add_argument("--dataset-original-path", required=True)
    token_distance.add_argument("--dataset-perturbed-path", required=True)
    token_distance.add_argument("--model-folder-path", default=MODEL_FOLDER_PATH)
    token_distance.add_argument("--emb-layer", type=int, default=None)
    token_distance.add_argument("--batch-size", type=int, default=128)
    token_distance.add_argument("--num-workers", type=int, default=12)
    token_distance.add_argument("--pin-memory", action="store_true")
    token_distance.add_argument(
        "--loss",
        choices=["sinkhorn", "energy", "gaussian"],
        default="sinkhorn",
    )
    token_distance.add_argument("--p", type=int, default=1)
    token_distance.add_argument("--blur", type=float, default=0.01)
    token_distance.add_argument("--backend", default="tensorized")
    token_distance.add_argument("--device", default=None)
    token_distance.add_argument("--keep-special-tokens", action="store_true")
    token_distance.add_argument("--output-path", default=None)

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "infer_token_distance":
        result = run_infer_token_distance(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    save_result(result, args.output_path)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#--model-folder-path '/nfs/team361/sb75/nichejepa-reproducibility/artifacts/models/06022026_161017_190'
#--model-folder-path /nfs/team361/sb75/nichejepa-reproducibility/artifacts/models/01112025_081731_614
'''
w1
python3 simple_infer_runner.py infer_token_distance \
  --dataset-original-path /nfs/team361/sb75/DATASETS/gold/cell-graph-tokenizer/kidney_perturbation/tokenized_data/nemokidneyxeniumatlas_unperturbed_case14 \
  --dataset-perturbed-path /nfs/team361/sb75/DATASETS/gold/cell-graph-tokenizer/kidney_perturbation/tokenized_data/nemokidneyxeniumatlas_perturbed_case14 \
  --model-folder-path /nfs/team361/sb75/nichejepa-reproducibility/artifacts/models/06022026_161017_190 \
  --loss sinkhorn \
  --p 1 \
  --blur 0.01 --batch-size 700\
  --output-path token_distance_w1.pkl


w2

python3 simple_infer_runner.py infer_token_distance \
  --dataset-original-path /nfs/team361/sb75/DATASETS/gold/cell-graph-tokenizer/kidney_perturbation/tokenized_data/nemokidneyxeniumatlas_unperturbed_case14 \
  --dataset-perturbed-path /nfs/team361/sb75/DATASETS/gold/cell-graph-tokenizer/kidney_perturbation/tokenized_data/nemokidneyxeniumatlas_perturbed_case14 \
  --model-folder-path /nfs/team361/sb75/nichejepa-reproducibility/artifacts/models/01112025_081731_614 \
  --loss sinkhorn \
  --p 2 \
  --blur 0.01 \
  --output-path /nfs/team361/mv10/nemo_v9/token_distance_w2.pkl

energy
python3 simple_infer_runner.py infer_token_distance \
  --dataset-original-path /nfs/team361/sb75/DATASETS/gold/cell-graph-tokenizer/kidney_perturbation/tokenized_data/nemokidneyxeniumatlas_unperturbed_case14 \
  --dataset-perturbed-path /nfs/team361/sb75/DATASETS/gold/cell-graph-tokenizer/kidney_perturbation/tokenized_data/nemokidneyxeniumatlas_perturbed_case14 \
  --model-folder-path /nfs/team361/sb75/nichejepa-reproducibility/artifacts/models/06022026_161017_190 \
  --loss energy \
  --blur 0.5 --batch-size 700 \
  --output-path token_distance_energy.pkl

mmd
python3 -m pdb simple_infer_runner.py infer_token_distance \
  --dataset-original-path /nfs/team361/sb75/DATASETS/gold/cell-graph-tokenizer/kidney_perturbation/tokenized_data/nemokidneyxeniumatlas_unperturbed_case14 \
  --dataset-perturbed-path /nfs/team361/sb75/DATASETS/gold/cell-graph-tokenizer/kidney_perturbation/tokenized_data/nemokidneyxeniumatlas_perturbed_case14 \
  --model-folder-path /nfs/team361/sb75/nichejepa-reproducibility/artifacts/models/01112025_081731_614 \
  --loss gaussian \
  --blur 0.5 --batch-size 768 \
  --output-path token_distance_mmd_01112025_081731_614.pkl
'''
