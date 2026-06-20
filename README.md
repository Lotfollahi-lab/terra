# TERRA

## Installation

TERRA is published on PyPI as `terra-st` (the bare `terra` name was already
taken; the import name is still `terra`):

```shell
pip install terra-st
```

For a development install from a clone of this repository:

```shell
pip install -e .
```

### PyTorch / GPU note

TERRA depends on [PyTorch](https://pytorch.org). A plain `pip install terra-st`
pulls the **default** PyTorch wheel from PyPI, which on Linux is a CUDA build —
and that build must match your machine's NVIDIA driver. If the bundled CUDA is
newer than your driver, CUDA fails to initialize at runtime with an error like:

```text
RuntimeError: The NVIDIA driver on your system is too old (found version 12040).
```

PyPI cannot host the CUDA-specific PyTorch wheels (they live on
`download.pytorch.org`), so the CUDA build is an **install-time** choice. Install
the PyTorch build for your hardware **first**, then install TERRA:

```shell
# GPU — pick the CUDA version that matches your driver (see below):
pip install torch --index-url https://download.pytorch.org/whl/cu124
# ...or CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install terra-st
```

Find your driver's maximum supported CUDA version in the top-right of
`nvidia-smi` ("CUDA Version"); the installed `torch.version.cuda` must be **≤**
that value. Verify the install with:

```shell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Repository Structure
1. **`main.py`**  
   The main entry point for the project, which supports running training and evaluation sweeps. It includes command-line arguments for customization and handles multi-GPU setups.

2. **`configs/$DATASET.yaml`**  
   Configuration file that defines the dataset specific hyperparameters and settings used during the training process, such as model architecture, data handling, and optimization settings (```$DATASET``` is the name of the dataset, e.g. ```merfish_300k```).

3. **`src/terra/models/gene_transformer.py`**  
   Contains the model definition for the gene transformer, implementing the core architecture that will be trained and evaluated.

4. **`src/terra/train.py`**  
   Handles the training process in a distributed setting. This script contains the logic for executing the training loop and logging results.

5. **`src/terra/infer.py`**  
   Manages the evaluation process. It evaluates the trained model on the specified tasks and logs the performance metrics.

6. **`src/terra/utils/config.py`**  
   Includes helper functions to setup the model and batch size params.

7. **`src/terra/utils/embedding.py`**  
   Provides utility functions for handling and loading embeddings required by the model during training and inference.

8. **`src/terra/utils/evaluation.py`**  
   Includes helper functions to streamline the evaluation process, such as metrics calculations and data preparation.

9. **`src/terra/datasets/cell_neighborhood_dataset.py`**
   Includes helper functions to create torch datasets for data loading.

10. **`tests`**  
   Includes test cases for different functionalities.

## Usage

### Training

To start training with a single GPU, use the following command:

```shell
python -m pdb main.py --fname configs/$DATASET.yaml --devices cuda:0
```
where ```$DATASET``` is the name of the dataset, e.g. ```merfish_300k```.

To start training with multiple GPUs, use the following command:

```shell
python -m pdb main.py --fname configs/$DATASET.yaml --devices cuda:0 cuda:1
```

To perform a sweep during training, use:

```shell
python -m pdb main.py --fname configs/$DATASET.yaml --devices cuda:0 --do_sweep
```

For multi-node training, first configure the required settings in your job_config file. 
Then, execute the following command:

```shell
bsub_mn_mg_yaml configs/job/hst_corpus_70m_test.yaml
```