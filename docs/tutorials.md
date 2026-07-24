# Tutorials

End-to-end notebooks for applying a pretrained TERRA model to your own spatial
data. Running them requires an NVIDIA GPU.

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quickstart: Single Sample
:link: notebooks/zero_shot_quickstart
:link-type: doc

Embed one spatial sample with a pretrained model and identify cell types and
spatial niches.
:::

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` Quickstart: Multiple Samples
:link: notebooks/zero_shot_quickstart_multisample
:link-type: doc

Embed several sections/donors together and integrate them into a shared space.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Downstream Analysis
:link: notebooks/downstream_analysis
:link-type: doc

Gene-level embeddings, spatial gene-pair scoring, spatial structure (Earth Mover's Distance),
subsetting, and in-silico perturbation.
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` In-silico Perturbation
:link: notebooks/perturbation_tutorial
:link-type: doc

Knock out genes in silico across a section and quantify the effect on each niche
with a Wasserstein (W2) distance between unperturbed and perturbed embeddings.
:::

:::{grid-item-card} {octicon}`location;1.5em;sd-mr-1` Spatial Mapping of Perturbation Effects
:link: notebooks/spatial_mapping_tutorial
:link-type: doc

Compute a per-cell perturbation score with `infer_token_distance` and project it
back onto the tissue to localise where a knockout has its effect.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Supervised Finetuning
:link: notebooks/supervised_finetuning
:link-type: doc

Adapt a pretrained model to a classification task via selective weight
unfreezing, with LoRA available as an alternative.
:::

::::

```{toctree}
:hidden: true
:maxdepth: 1

notebooks/zero_shot_quickstart
notebooks/zero_shot_quickstart_multisample
notebooks/downstream_analysis
notebooks/perturbation_tutorial
notebooks/spatial_mapping_tutorial
notebooks/supervised_finetuning
```
