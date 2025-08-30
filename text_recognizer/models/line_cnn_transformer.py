import argparse
from typing import Any, Dict
import math
import torch
import torch.nn as nn

from .line_cnn import LineCNN
from .transformer_util import PositionalEncoding, generate_square_subsequent_mask


TF_DIM = 256
TF_FC_DIM = 256
TF_DROPOUT = 0.4
TF_LAYERS = 4
TF_NHEAD = 4