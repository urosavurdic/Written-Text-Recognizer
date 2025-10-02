from pathlib import Path
from typing import Dict, Collection, Optional, Tuple, Union
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from text_recognizer import util
from text_recognizer.data.util import BaseDataset


BATCH_SIZE = 128
NUM_WORKERS = 0
class BaseDataModule(pl.LightningDataModule):
    """
    Base class for data modules in PyTorch Lightning. 
    It provides a structure for loading datasets, applying transformations, and preparing data loaders.
    Args:
        args (argparse.Namespace, optional): Command line arguments for configuration.
        batch_size (int, optional): Batch size for training and validation. Defaults to 128.
        num_workers (int, optional): Number of workers for data loading. Defaults to 0.
    Attributes:
        args (Dict[str, Any]): Dictionary of command line arguments.
        batch_size (int): Batch size for training and validation.
        num_workers (int): Number of workers for data loading.
        dim (Tuple[int, int]): Dimensions of the input data (height, width).
        output_dim (int): Dimension of the output data, typically the number of classes.
        char_to_idx (Dict[str, int]): Mapping from characters to their corresponding indices.
    """
    def __init__(self, args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get('batch_size', BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # to be set in subclasses
        self.dim: Tuple[int, ...]
        self.output_dim: Tuple[int, ...]
        self.char_to_idx: Collection
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]

    @classmethod
    def data_directory_path(cls):
        """
        Returns the path to the directory where the dataset is stored.
        """
        return Path(__file__).resolve().parents[1] / 'data' / 'datasets'
    
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        """
        Adds command line arguments to the parser.
        Args:
            parser (argparse.ArgumentParser): The argument parser to which arguments will be added.
        """
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training and validation')
        parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='Number of workers for data loading')

        return parser    
    
    def configuration(self):
        """
        Returns the configuration of the data module as a dictionary.
        Returns:
            Dict[str, Any]: Configuration dictionary containing batch size, number of workers, dimensions, and character-to-index mapping.
        """
        
        return {
            'input_dim': self.dim,
            'output_dim': self.output_dim,
            'char_to_idx': self.char_to_idx,
        }
    
    def prepare_data(self):
        """
        Prepares the data by downloading or processing it if necessary.
        This method is called only once and is used to set up the dataset.
        """
        pass

    def setup(self, stage: str = None):
        """
        Sets up the data for training, validation, and testing.
        """
        pass

    def train_dataloader(self):
        """
        Returns:
            DataLoader: Data loader for the training dataset.
        """
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)
    
    def val_dataloader(self):
        """
        Returns:
            DataLoader: Data loader for the validation dataset.
        """
        return DataLoader(self.data_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)
    
    def test_dataloader(self):
        """
        Returns:
            DataLoader: Data loader for the test dataset.
        """
        return DataLoader(self.data_test, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)

def load_and_print_info(data_module: type) -> None:

    """
    Loads the dataset and prints its configuration.
    """
    
    parser = argparse.ArgumentParser()
    data_module.add_arguments(parser)
    args = parser.parse_args()
    dataset = data_module(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)


def _download_raw_data(metadata: Dict, dl_dirname: Path) -> Path:
    
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata['filename']
    if filename.exists():
        print(f"File {filename} already exists. Skipping download.")
        return filename
    print(f"Downloading {metadata['url']} to {filename}")
    util.download_url(metadata['url'], filename)
    print("Computing sha256 checksum...")
    sha256 = util.compute_sha256(filename)
    if sha256 != metadata['sha256']:
        raise ValueError(f"Checksum mismatch for {filename}. Expected {metadata['sha256']}, got {sha256}.")
    return filename