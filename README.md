```markdown
# Nichejepa

## Installation

To install the project and its dependencies, run:

```shell
pip install -e .
```

## Repository Structure

1. **`main_sweep.py`**  
   The main entry point for the project, which supports running training and evaluation sweeps. It includes command-line arguments for customization and handles multi-GPU setups.

2. **`configs/cnd_gtb10_ep300.yaml`**  
   This configuration file defines the hyperparameters and settings used during the training process, such as model architecture, data handling, and optimization settings.

3. **`src/nichejepa/models/gene_transformer.py`**  
   Contains the model definition for the gene transformer, implementing the core architecture that will be trained and evaluated.

4. **`src/nichejepa/train_sweep.py`**  
   Handles the training process in a distributed setting. This script contains the logic for executing the training loop and logging results.

5. **`src/nichejepa/eval_sweep.py`**  
   Manages the evaluation process. It evaluates the trained model on the specified tasks and logs the performance metrics.

6. **`src/nichejepa/utils/emb_utils.py`**  
   Provides utility functions for handling and loading embeddings required by the model during training and inference.

7. **`src/nichejepa/utils/eval_utils.py`**  
   Includes helper functions to streamline the evaluation process, such as metrics calculations and data preparation.

## Usage

### Training

To start training, use the following command:

```shell
python -m pdb main_sweep.py --fname configs/cnd_gtb10_ep300.yaml --task cell_type --devices cuda:0 --test
```

### Training with Sweep

To perform a sweep during training, use:

```shell
python -m pdb main_sweep.py --fname configs/cnd_gtb10_ep300.yaml --task cell_type --devices cuda:0 --test --do_sweep
```
```

You can copy and paste this markdown code into your README file on GitHub. It will render correctly with the appropriate formatting for sections, code blocks, and bullet points.
