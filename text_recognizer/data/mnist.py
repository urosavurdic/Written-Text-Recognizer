import argparse
import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST as TorchMNIST
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_directory_path() / "downloaded"

class MNIST(BaseDataModule):
    """
    Data module for the MNIST dataset. Inherits from BaseDataModule.
    """

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.data_dirname = DOWNLOADED_DATA_DIRNAME / "mnist"
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dim = (1, 28, 28)  # MNIST images are grayscale and 28x28 pixels
        self.output_dim = (1,) # MNIST has a single output dimension (the digit)
        self.num_classes = 10 # specify output classes
        self.char_to_idx = {'<S>': 0, '<E>': 1, '<P>': 2} # Start, End, and Padding tokens.
    
    @staticmethod
    def add_data_specific_args(parser):
        """
        Add MNIST-specific arguments to the parser.
        """
        # No MNIST-specific arguments for now, just return the parser.
        return parser
    
    def prepare_data(self):
        """
        Prepares the MNIST dataset by downloading it if it is not already present.
        This method is called only once and is used to set up the dataset.
        """
        TorchMNIST(self.data_dirname, train=True, download=True)
        TorchMNIST(self.data_dirname, train=False, download=True)
    
    def setup(self, stage: str = None):
        """
        Sets up the MNIST dataset for training, validation, and testing.
        """
        full_train = TorchMNIST(self.data_dirname, train=True, transform=self.transform)
        self.data_train, self.data_val = torch.utils.data.random_split(full_train, [55000, 5000])
        self.data_test = TorchMNIST(self.data_dirname, train=False, transform=self.transform)
    

if __name__ == "__main__":
    load_and_print_info(MNIST)

