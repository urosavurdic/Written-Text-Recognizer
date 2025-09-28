from typing import Any, Dict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


CONV_DIM = 64
FC_DIM = 128
IMG_SIZE = 28

class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(output_channels) # to stabilize training and improves generalization and ensure good gradient flow
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x (tensor of B, C, H, W dimensions)
        Output:
            torch.Tensor (B, C, H, W dimensions)

        """

        c = self.conv(x)
        b = self.bn(c)
        r = self.relu(b)

        return r
    
class CNN(nn.Module):
    """
    Simple CNN for recognizing characters in a square image
    """
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        input_dim = data_config["input_dim"]
        output_dim = data_config["output_dim"]
        num_classes = len(data_config["chr)_to_idx"])

        if isinstance(output_dim, tuple):
            output_dim = output_dim[0]

        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)

        self.conv1 = ConvBlock(input_dim[0], conv_dim)
        self.conv2 = ConvBlock(conv_dim, 2*conv_dim)
        self.conv3 = ConvBlock(2*conv_dim, 2*conv_dim)
        self.conv4 = ConvBlock(2*conv_dim, conv_dim)
        self.dropout = nn.Dropout(0.4)
        self.max_pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1) # to reduce overfitting and make network more efficient reduces HxW to 1x1

        """
        3x3 convs have padding size 1 => leaves input size unchanged
        2x2 max pooling => divides input size by 2
        flattening => squares it
        """
        """
        conv_output_size = IMG_SIZE // 2
        fc_input_dim = int(conv_output_size * conv_output_size * conv_dim)
        """
        self.fc1 = nn.Linear(conv_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x (tensor of B, C, H, W dimensions), where H == W == IMG_SIZE
        Output:
            torch.Tensor (B, C dimensions)
        """
        _B, _C, H, W = x.shape
        assert H == W == IMG_SIZE

        # feature extractor
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.max_pool(x)
        x = self.gap(x) # [B, C, 1, 1]
        x = torch.flatten(x, 1) # [B, C]

        # classifier head
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)

        return parser
