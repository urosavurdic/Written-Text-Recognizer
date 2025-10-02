import argparse
import sys
import shutil
import json
from pathlib import Path
import tempfile
from typing import Optional, Union
import wandb


FILE_NAME = Path(__file__).resolve()
ARTIFACTS_BASE_DIRNAME = FILE_NAME.parents[1] / "text_recognizer" / "artifacts"
TRAINING_LOGS_DIRNAME = FILE_NAME.parent / "logs"

def save_best_model():
    pass

if __name__ == "__main__":
    save_best_model()