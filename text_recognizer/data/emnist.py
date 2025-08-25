"""
Downloads EMNIST dataset if not already present from NIST website as .npz files.
"""

import os
import numpy as np
import json
from pathlib import Path
import zipfile
from typing import Sequence
from torchvision import transforms
import h5py
import toml
import shutil
from scipy.io import loadmat

from text_recognizer.data.base_data_module import BaseDataModule, _download_raw_data, load_and_print_info
from text_recognizer.data.util import BaseDataset, split_dataset

NUM_SPECIAL_TOKENS = 4
SAMPLE_TO_BALANCE = True  # If true, take at most the mean number of instances per class.
TRAIN_FRAC = 0.8

RAW_DATA_DIRNAME = BaseDataModule.data_directory_path() / "raw" / "emnist"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_directory_path() / "downloaded" / "emnist"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_directory_path() / "processed" / "emnist"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "byclass.h5"
ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "emnist_essentials.json"




class EMNIST(BaseDataModule):
    """
    EMNIST dataset class for loading and processing the EMNIST dataset. 
    The EMNIST dataset is a set of handwritten character digits and letters.
    It is a subset of the NIST Special Database 19, converted to a format 28x28 pixels that is compatible with the MNIST dataset.
    https://www.nist.gov/itl/products-and-services/emnist-dataset
    The data split we use is the 'ByClass': 62 unbalanced classes (10 digits + 26 uppercase + 26 lowercase letters).
    """
    def __init__(self, args = None):
        super().__init__(args)
        
        if not os.path.exists(ESSENTIALS_FILENAME):
            _download_and_process_emnist() # Download and process the EMNIST dataset if not already done.
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f) # Load the essentials from the JSON file

        self.char_to_idx = list(essentials['char_to_idx']) # Convert the character to index mapping to a list
        self.inverse_char_to_idx = {v: k for k, v in enumerate(self.char_to_idx)} # inverse mapping for quick lookup
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]) # Normalize the images to have mean 0.1307 and std 0.3081

        self.dim = (1, *essentials['input_dim']) # extra dimension are added by the transforms
        self.output_dim = (1,) # EMNIST has a single output dimension (the character)
        self.num_classes = 62 + NUM_SPECIAL_TOKENS
    
    @staticmethod
    def add_data_specific_args(parser):
        """
        Add EMNIST-specific arguments to the parser.
        Right now we don't need any, so just return the parser.
        """
        return parser

    def prepare_data(self, *args, **kwargs):
        """
        Prepares the EMNIST dataset by downloading and processing it if not already done.
        inputs:
            *args, **kwargs: Additional arguments (not used).
        """
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_process_emnist()
        with open(ESSENTIALS_FILENAME) as f:
            _essentials = json.load(f)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, 'r') as f:
                self.x_train_val = f['x_train'][:]
                self.y_train_val = f['y_train'][:].squeeze().astype(int)

            data_train_val = BaseDataset(self.x_train_val, self.y_train_val, transform=self.transform)
            self.data_train, self.data_val = split_dataset(data_train_val, TRAIN_FRAC, seed=42)


        if stage == 'test' or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, 'r') as f:
                self.x_test = f['x_test'][:]
                self.y_test = f['y_test'][:].squeeze().astype(int)
            self.data_test = BaseDataset(self.x_test, self.y_test, transform=self.transform)
    
    def __repr__(self):
        basic = f"EMNIST(num_classes={len(self.char_to_idx)}, dim={self.dim}, output_dim={self.output_dim})"
        if self.train_dataset is None and self.val_dataset is None and self.test_dataset is None:
            return basic
    

    
### Helper functions to download and process the EMNIST dataset

def _download_and_process_emnist():
    """
    Downloads and processes the EMNIST dataset, saving it in a processed format.
    """
    metadata = toml.load(METADATA_FILENAME)
    _download_raw_data(metadata, DL_DATA_DIRNAME)
    _process_raw_data(metadata["filename"], DL_DATA_DIRNAME)

def _process_raw_data(filename: str, data_dirname: Path):
    """
    Processes the raw EMNIST data from the downloaded .npz file and saves it in a compressed HDF5 format.
    This function extracts the images and labels, balances the dataset if specified, and saves the processed data.
    inputs:
        filename: str - The name of the downloaded zip file containing the EMNIST dataset.
        data_dirname: Path - The directory where the downloaded file is located.
    returns:
        None
    """
    print("Unzipping EMNIST data...")
    curdir = os.getcwd()
    os.chdir(data_dirname)
    zip_file = zipfile.ZipFile(filename, 'r')
    zip_file.extract("matlab/emnist-byclass.mat") # Extract the .mat file from the zip

    print("Loading .mat file...")
    data = loadmat("matlab/emnist-byclass.mat")
    # Extract images and labels from the loaded data
    x_train = data['dataset']['train'][0, 0]['images'][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_train = data['dataset']['train'][0, 0]['labels'][0, 0] + NUM_SPECIAL_TOKENS
    x_test = data['dataset']['test'][0, 0]['images'][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_test = data['dataset']['test'][0, 0]['labels'][0, 0] + NUM_SPECIAL_TOKENS

    # NOTE: We add NUM_SPECIAL_TOKENS to the labels to account for the special tokens in the character set.

    if SAMPLE_TO_BALANCE:
        # Sample to balance the dataset
        print("Balancing classes...")
        x_train, y_train = _sample_to_balance(x_train, y_train)
        x_test, y_test = _sample_to_balance(x_test, y_test)

    # Convert the labels to uint8 for compatibility with HDF5.
    print("Saving HDF5 in compressed format...")
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    with h5py.File(PROCESSED_DATA_FILENAME, 'w') as f:
        f.create_dataset('x_train', data=x_train, dtype='u1', compression='lzf')
        f.create_dataset('y_train', data=y_train, dtype='u1', compression='lzf')
        f.create_dataset('x_test', data=x_test, dtype='u1', compression='lzf')
        f.create_dataset('y_test', data=y_test, dtype='u1', compression='lzf')
    

    print("Saving essentials...")
    # Save the character to index mapping and input dimensions in a JSON file.
    print("Saving essentials...")
    mapping = data['dataset']['mapping'][0, 0]
    char_to_idx = {int(m[0]): chr(m[1]) for m in mapping}

    essentials = {
        'char_to_idx': char_to_idx,          
        'input_dim': list(x_train.shape[1:])
    }
    with open(ESSENTIALS_FILENAME, 'w') as f:
        json.dump(essentials, f)   
        
        print("Cleaning up...")
        # Clean up the downloaded files
        shutil.rmtree("matlab")
        os.chdir(curdir)
        print("EMNIST dataset processed and saved successfully.")

def _sample_to_balance(x, y):
    """
    Samples the dataset to balance the classes by limiting the number of samples per class to the mean number of samples across all classes.
    If a class has fewer samples than the mean, all its samples are preserved.
    inputs:
        x: np.ndarray of shape (N, H, W) - images
        y: np.ndarray of shape (N, 1) - labels
    returns:
        x_sampled: np.ndarray of shape (M, H, W) - balanced images
        y_sampled: np.ndarray of shape (M, 1) - balanced labels
    """
    np.random.seed(42)  # For reproducibility
    num_to_sample = int(np.bincount(y.flatten()).mean())
    all_indices = []
    for label in np.unique(y.flatten()):
        indices = np.where(y == label)[0]
        sampled_indices = np.unique(np.random.choice(indices, num_to_sample))
        all_indices.append(sampled_indices)
    
    indices = np.concatenate(all_indices)
    x_sampled = x[indices]
    y_sampled = y[indices]

    return x_sampled, y_sampled

def _augment_emnist_characters(characters: Sequence[str]) -> Sequence[str]:
    """
    Augments the EMNIST character set with extra characters from the IAM dataset. 
    """
    iam_characters = [
        " ",
        "!",
        '"',
        "#",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "?",
    ]

    return ['<B>', '<S>', '<E>', '<P>', *characters, *iam_characters]
    # <B> - CTC blank token
    # <S> - start of sequence token
    # <E> - end of sequence token
    # <P> - padding token

if __name__ == "__main__":
    load_and_print_info(EMNIST)