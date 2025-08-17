from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

FC1_SIZE = 1024
FC2_SIZE = 128

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model for text recognition tasks.
    This model consists of two fully connected layers with ReLU activations and dropout for regularization.
    """

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None):
        super().__init__()
        self.args = args if args is not None else {}

        input_dim = np.prod(data_config['input_dim'])
        num_classes = len(data_config['char_to_idx'])

        fc1_dim = self.args.get('fc1', FC1_SIZE)
        fc2_dim = self.args.get('fc2', FC2_SIZE)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the MLP model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        """
        Adds command line arguments specific to the MLP model.
        Args:
            parser (argparse.ArgumentParser): The argument parser to which arguments will be added.
        """
        parser.add_argument('--fc1', type=int, default=FC1_SIZE, help='Size of the first fully connected layer')
        parser.add_argument('--fc2', type=int, default=FC2_SIZE, help='Size of the second fully connected layer')
        return parser


