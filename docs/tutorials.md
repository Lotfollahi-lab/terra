# Tutorials

Two end-to-end notebooks walk through applying a pretrained TERRA model to your
own spatial data. They require an NVIDIA GPU and ship with outputs cleared — run
them on a GPU to reproduce the figures.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Zero-shot Quickstart
:link: notebooks/zero_shot_quickstart
:link-type: doc

Embed a spatial dataset with a pretrained model and identify cell types and
spatial niches.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Downstream Analysis
:link: notebooks/downstream_analysis
:link-type: doc

Gene-level embeddings, spatial gene-pair scoring, EMD-based spatial structure,
subsetting, and in-silico perturbation.
:::

::::

```{toctree}
:hidden: true
:maxdepth: 1

notebooks/zero_shot_quickstart
notebooks/downstream_analysis
```
