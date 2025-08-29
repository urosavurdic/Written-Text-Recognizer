from typing import Any, Dict
import argparse
import math

import torch
import torch.nn as nn
from .cnn import CNN, IMG_SIZE

WINDOW_WIDTH = 28
WINDOW_STRIDE = 28

class LineCNNSimple(nn.Module):
    def __init__(self, data_config: Dict[Any, Any], args:argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.WW = self.args.get("window_width", WINDOW_WIDTH)
        self.WS = self.args.get("window_stride", WINDOW_STRIDE)

        self.limit_output_length = self.args.get("limit_output_length", False)

        self.num_classes = len(data_config["char_to_idx"])

        self.output_length = data_config["output_dim"][0]
        self.cnn = CNN(data_config=data_config, args=args)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x (B, C, H, W): imput image:
                B - Batch size
                C - Num of channels (1 in this case because of grayscale)
                H - Hight
                W - Width
        Returns:
            (B, C, S):
                B - Batch size
                C - number of classes
                S - Sequence Length: computed from W and CHAR_WIDTH
        """
        B, _C, H, W = x.shape #_C is not used directlly
        assert H == IMG_SIZE
        S = math.floor((W-self.WW)/self.WS + 1)
        activations = torch.zeros((B, self.num_classes, S)).type_as(x)
        for s in range(S):
            start_w = self.WS * s
            end_w = start_w + self.WW
            window = x[:, :, :, start_w:end_w] # -> (B, C, H, self.WW)
            activations[:, :, s] = self.cnn(window)
        
        if self.limit_output_length:
            # S might not match ground truth, so let's only take enough activations as are expected
            activations = activations[:, :, : self.output_length]
        
        return activations
    
    @staticmethod
    def add_to_argparse(parser):
        CNN.add_to_argparse(parser)
        parser.add_argument("--window_width", type=int, default=WINDOW_WIDTH, help="Width of the window that will slide over the input image.")
        parser.add_argument("--window_stride", type=int, default=WINDOW_STRIDE, help="Stride of the window that will slide over the input image.")
        parser.add_argument("--limit_output_length", action="store_true", default=False)
        
        return parser