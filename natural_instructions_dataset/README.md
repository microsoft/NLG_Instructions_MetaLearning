# Experiments with Natural Language Instructions dataset

## Setting up the Data Directory
- Download data from repository: allenai/natural-instructions-expansion: Expanding natural instructions. https://github.com/allenai/natural-instructions-expansion
- Run data_utils.py for some basic splits and preprocessing of the NI dataset

## Training

To train a model using the NLI dataset, use the src/main_driver.py script, or execute the script `train.sh` from the main directory as follows:

    $ bash data/natural_instructions_dataset/train.sh <input_dirpath> <output_dirpath> <cuda_device>