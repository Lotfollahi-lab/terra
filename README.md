# Nichejepa

## Installation

To install the project and its dependencies, run:

```shell
pip install -e .
```

## Repository Structure
It contains most important files.
1. **`main.py`**  
   The main entry point for the project, which supports running training and evaluation sweeps. It includes command-line arguments for customization and handles multi-GPU setups.

2. **`configs/cnd_gtb10_ep300.yaml`**  
   This configuration file defines the hyperparameters and settings used during the training process, such as model architecture, data handling, and optimization settings.

3. **`src/nichejepa/models/gene_transformer.py`**  
   Contains the model definition for the gene transformer, implementing the core architecture that will be trained and evaluated.

4. **`src/nichejepa/train.py`**  
   Handles the training process in a distributed setting. This script contains the logic for executing the training loop and logging results.

5. **`src/nichejepa/eval.py`**  
   Manages the evaluation process. It evaluates the trained model on the specified tasks and logs the performance metrics.

6. **`src/nichejepa/utils/emb_utils.py`**  
   Provides utility functions for handling and loading embeddings required by the model during training and inference.

7. **`src/nichejepa/utils/eval_utils.py`**  
   Includes helper functions to streamline the evaluation process, such as metrics calculations and data preparation.

8. **`src/nichejepa/utils/config_utils.py`**  
   Includes helper functions to setup the model and batch size params.

9. **`src/nichejepa/datasets/cell_neighborhood_dataset.py`**
   Includes helper functions to create datasets.

10. **`tests`**  
   Includes test cases for differnet functions.

## Usage

### Training

To start training, use the following command:

```shell
python -m pdb main.py --fname configs/cnd_gtb10_ep300.yaml --devices cuda:0 
```

### Training with Sweep

To perform a sweep during training, use:

```shell
python -m pdb main.py --fname configs/cnd_gtb10_ep300.yaml --devices cuda:0 --do_sweep
```


