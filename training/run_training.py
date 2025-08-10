import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
import wandb

from text_recognizer import lit_models

np.random.seed(42)
torch.manual_seed(42)
