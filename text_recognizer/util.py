"""
Utility functions for text recognition tasks.
Contains functions for reading and writing images, computing SHA256 hashes, and converting class vectors to binary class matrices.
Also includes a custom tqdm progress bar for tracking file downloads.
"""
from io import BytesIO
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve
import base64
import hashlib

from PIL import Image
from tqdm import tqdm
import numpy as np
import smart_open


def to_categorical(y, num_classes):
    """1-hot encode a tensor."""
    return np.eye(num_classes, dtype="uint8")[y]


def read_image_pil(image_uri: Union[Path, str], grayscale=False) -> Image:
    with smart_open.open(image_uri, "rb") as image_file:
        return read_image_pil_file(image_file, grayscale)


def read_image_pil_file(image_file, grayscale=False) -> Image:
    with Image.open(image_file) as image:
        if grayscale:
            image = image.convert(mode="L")
        else:
            image = image.convert(mode=image.mode)
        return image

def compute_sha256(filename: Union[Path, str]):
    """Return SHA256 checksum of a file."""
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

class TqdmUpTo(tqdm):
    """
    A tqdm progress bar that can be used to track the progress of file downloads.
    Parameters:
        b (int): Current block number.
        bsize (int): Size of each block (in bytes).
        tsize (int, optional): Total size of the file (in bytes). If None, the total size is unknown.
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Updates the progress bar.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
    
    def download_url(self, url, filename):
        """
        Downloads a file from a URL and shows progress.
        """
        filename_str = str(filename)
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename_str) as t:
            urlretrieve(url, filename_str, reporthook=t.update_to, data=None)

def download_url(url, filename):
    downloader = TqdmUpTo()
    downloader.download_url(url, filename)

    