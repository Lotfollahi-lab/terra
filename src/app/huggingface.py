"""Hugging Face Hub integration for TERRA models.

Publish and download TERRA model *bundles* -- the self-contained folder that the
inference pipeline (``app.inference.embed_dataset`` /
``harmonize_tokenize_embed_pipeline``) reads via ``model_folder_path``:

    model_checkpoint.pt     target-encoder weights (inference)
    model_config.yaml       model / tokenization config
    token_dictionary.pkl    gene-token vocabulary
    ensembl_dic.pkl         gene-name -> Ensembl-ID mapping (harmonization)
    norm_factors.csv        (optional) frozen gene-level norm factors
    pf_targets.csv          (optional) frozen PFlog1pPF targets

Model family layout on the Hub: one repo per named model
(``Lotfollahi-lab/TERRA-96M``, ``Lotfollahi-lab/TERRA-<next>``, ...), and git
tags (``revision=``) for versions of the *same* model -- tag the manuscript
checkpoint (e.g. ``v1.0``) so it can be cited as an immutable revision.

Requires the optional ``huggingface_hub`` dependency::

    pip install "terra-st[hub]"

Examples
--------
Upload a trained model bundle (maintainer)::

    python -m app.huggingface upload \\
        --folder /path/to/artifacts/models/<timestamp> \\
        --repo-id Lotfollahi-lab/TERRA-96M \\
        --corpus HST-Corpus-110M --tag v1.0

Download a published model and run inference (user)::

    from app.huggingface import download_pretrained
    from app.inference import harmonize_tokenize_embed_pipeline

    d = download_pretrained("Lotfollahi-lab/TERRA-96M", revision="v1.0")
    adata = harmonize_tokenize_embed_pipeline(
        adata=adata, model_folder_path=d,
        gene_mapping_dict_file_path=f"{d}/ensembl_dic.pkl", ...)
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Files that make up a self-contained TERRA inference bundle. Optional files
# that are absent for a given model are simply skipped on upload.
BUNDLE_FILES = (
    "model_checkpoint.pt",
    "model_config.yaml",
    "token_dictionary.pkl",
    "ensembl_dic.pkl",
    "norm_factors.csv",
    "pf_targets.csv",
    "README.md",
)

DEFAULT_LICENSE = "cc-by-nc-4.0"


def _require_hub():
    """Import ``huggingface_hub`` lazily with an actionable error if missing."""
    try:
        import huggingface_hub
    except ModuleNotFoundError as e:  # pragma: no cover - trivial guard
        raise ModuleNotFoundError(
            "app.huggingface requires 'huggingface_hub'. Install it with "
            '`pip install "terra-st[hub]"` (or `pip install huggingface_hub`).'
        ) from e
    return huggingface_hub


def build_model_card(repo_id: str,
                     corpus: str | None = None,
                     license: str = DEFAULT_LICENSE) -> str:
    """Return a Markdown model card (with YAML front matter) for a TERRA repo."""
    name = repo_id.split("/")[-1]
    corpus_line = f"Trained on **{corpus}**.\n\n" if corpus else ""
    header = f"""---
license: {license}
library_name: terra-st
tags:
- spatial-transcriptomics
- foundation-model
- jepa
- single-cell
pipeline_tag: feature-extraction
---
# {name}

JEPA-based spatial-transcriptomics foundation model (TERRA).
{corpus_line}Code & docs: https://github.com/Lotfollahi-lab/terra

## Files
- `model_checkpoint.pt` — target-encoder weights (inference)
- `model_config.yaml` — model / tokenization config
- `token_dictionary.pkl` — gene-token vocabulary
- `ensembl_dic.pkl` — gene-name to Ensembl-ID mapping (harmonization)

## Usage
"""
    # Kept as a plain string (not an f-string) so the ``{d}`` in the snippet is
    # literal; the repo id is substituted explicitly.
    usage = '''```python
from app.huggingface import download_pretrained
from app.inference import harmonize_tokenize_embed_pipeline

d = download_pretrained("__REPO_ID__")
adata = harmonize_tokenize_embed_pipeline(
    adata=adata,
    model_folder_path=d,
    gene_mapping_dict_file_path=f"{d}/ensembl_dic.pkl",
    # ... sample_key / batch_key / etc.
)
```

## Citation
<add paper / bioRxiv reference>
'''
    return header + usage.replace("__REPO_ID__", repo_id)


def push_model_to_hub(model_folder: str | Path,
                      repo_id: str,
                      *,
                      corpus: str | None = None,
                      private: bool = True,
                      tag: str | None = None,
                      license: str = DEFAULT_LICENSE,
                      model_card: str | None = None,
                      commit_message: str | None = None,
                      token: str | None = None) -> str:
    """Create (if needed) a HF model repo and upload a TERRA model bundle.

    Only the standard bundle files present in ``model_folder`` are uploaded
    (training cruft like optimizer state / logs / intermediate epochs is
    skipped). A model card is generated unless ``model_card`` is provided. If
    ``tag`` is given, an immutable git tag is created on the uploaded revision
    -- use it to pin the manuscript checkpoint (e.g. ``v1.0``).

    Returns the repository URL.
    """
    hub = _require_hub()
    model_folder = Path(model_folder)
    if not model_folder.is_dir():
        raise NotADirectoryError(f"{model_folder} is not a directory.")

    api = hub.HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    card = (model_card if model_card is not None
            else build_model_card(repo_id, corpus=corpus, license=license))
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add/update model card",
    )

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_folder),
        allow_patterns=list(BUNDLE_FILES),
        commit_message=commit_message or f"Upload TERRA bundle ({repo_id})",
    )

    if tag:
        api.create_tag(repo_id, tag=tag, repo_type="model", exist_ok=True)

    url = f"https://huggingface.co/{repo_id}"
    print(f"Uploaded {model_folder} -> {url}" + (f"  (tag: {tag})" if tag else ""))
    return url


def download_pretrained(repo_id: str,
                        *,
                        revision: str | None = None,
                        cache_dir: str | None = None,
                        token: str | None = None) -> str:
    """Download a published TERRA model bundle.

    Returns the local folder path to pass as ``model_folder_path`` to the
    inference pipeline. ``revision`` pins a git tag/branch/commit (e.g. the
    manuscript ``v1.0``); omit it for the latest ``main``.
    """
    hub = _require_hub()
    return hub.snapshot_download(
        repo_id,
        repo_type="model",
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m app.huggingface {upload,download} ...`` (or ``terra-hub``)."""
    p = argparse.ArgumentParser(
        description="Publish/download TERRA models on the Hugging Face Hub.")
    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="Upload a TERRA model bundle folder.")
    up.add_argument("--folder", required=True,
                    help="Path to the model bundle folder.")
    up.add_argument("--repo-id", required=True,
                    help="e.g. Lotfollahi-lab/TERRA-96M")
    up.add_argument("--corpus", default=None,
                    help="Training corpus for the model card, e.g. HST-Corpus-110M")
    up.add_argument("--tag", default=None,
                    help="Immutable git tag to create, e.g. v1.0")
    up.add_argument("--license", default=DEFAULT_LICENSE)
    up.add_argument("--public", action="store_true",
                    help="Make the repo public (default: private).")
    up.add_argument("--token", default=None,
                    help="HF write token (else cached login / HF_TOKEN env).")

    dl = sub.add_parser("download", help="Download a TERRA model bundle.")
    dl.add_argument("--repo-id", required=True)
    dl.add_argument("--revision", default=None, help="git tag/branch/commit.")
    dl.add_argument("--token", default=None)

    args = p.parse_args(argv)
    if args.cmd == "upload":
        push_model_to_hub(
            args.folder, args.repo_id, corpus=args.corpus,
            private=not args.public, tag=args.tag, license=args.license,
            token=args.token)
    elif args.cmd == "download":
        print(download_pretrained(
            args.repo_id, revision=args.revision, token=args.token))


if __name__ == "__main__":
    main()
