# TERRA

**TERRA** is a JEPA-based foundation model for spatial transcriptomics, developed by the
[Lotfollahi Lab](https://github.com/Lotfollahi-lab). It learns embeddings of cells in
their spatial context by predicting masked cell and neighbourhood representations in
latent space, and transfers to downstream tasks such as niche identification,
batch-integrated atlasing, spatial gene-pair scoring, and in-silico perturbation.

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} {octicon}`desktop-download;1.5em;sd-mr-1` Installation
:link: installation
:link-type: doc

Check out the installation guide to set up TERRA and PyTorch for your hardware.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Tutorials
:link: tutorials
:link-type: doc

Learn by following an end-to-end example application of TERRA.
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` User Guide
:link: user_guide
:link-type: doc

Understand the concepts, the inference pipeline, and the pretrained models.
:::

:::{grid-item-card} {octicon}`code-square;1.5em;sd-mr-1` API
:link: api
:link-type: doc

Detailed descriptions of TERRA's public functions and classes.
:::

:::{grid-item-card} {octicon}`tag;1.5em;sd-mr-1` Release Notes
:link: changelog
:link-type: doc

Follow the latest changes and version history.
:::

:::{grid-item-card} {octicon}`git-pull-request;1.5em;sd-mr-1` Contributing
:link: contributing
:link-type: doc

Learn how to contribute to the TERRA project.
:::

::::

If you find TERRA useful for your research, please consider citing the manuscript
(see the {doc}`user_guide`).

```{toctree}
:hidden: true
:maxdepth: 2

installation
tutorials
user_guide
api
changelog
contributing
references
```
