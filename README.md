# NicheJEPA

## Installation

To install the project and its dependencies, run:

```shell
pip install -e .
```

## Repository Structure
1. **`main.py`**  
   The main entry point for the project, which supports running training and evaluation sweeps. It includes command-line arguments for customization and handles multi-GPU setups.

2. **`configs/$DATASET.yaml`**  
   Configuration file that defines the dataset specific hyperparameters and settings used during the training process, such as model architecture, data handling, and optimization settings (```$DATASET``` is the name of the dataset, e.g. ```merfish_300k```).

3. **`src/nichejepa/models/gene_transformer.py`**  
   Contains the model definition for the gene transformer, implementing the core architecture that will be trained and evaluated.

4. **`src/nichejepa/train.py`**  
   Handles the training process in a distributed setting. This script contains the logic for executing the training loop and logging results.

5. **`src/nichejepa/infer.py`**  
   Manages the evaluation process. It evaluates the trained model on the specified tasks and logs the performance metrics.

6. **`src/nichejepa/utils/config.py`**  
   Includes helper functions to setup the model and batch size params.

7. **`src/nichejepa/utils/embedding.py`**  
   Provides utility functions for handling and loading embeddings required by the model during training and inference.

8. **`src/nichejepa/utils/evaluation.py`**  
   Includes helper functions to streamline the evaluation process, such as metrics calculations and data preparation.

9. **`src/nichejepa/datasets/cell_neighborhood_dataset.py`**
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