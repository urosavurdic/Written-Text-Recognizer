"""
Utility functions for text recognition tasks.
Contains functions for reading and writing images, computing SHA256 hashes, and converting class vectors to binary class matrices.
Also includes a custom tqdm progress bar for tracking file downloads.
"""
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Union
from urllib.request import urlopen, urlretrieve
import hashlib
import os

import numpy as np
import cv2
from tqdm import tqdm
from urllib.request import urlopen, urlretrieve

def to_categorical(y, num_classes):
    """
    Converts a class vector (integers) to binary class matrix.
    """
    return np.eye(num_classes, dtype='uint8')[y]

def read_image(image_uri: Union[str, Path], grayscale: bool = False) -> np.array:
    """"
    Reads an image from a file or URL.
    """
    def read_image_from_filename(image_filename, imread_flag):
        """
        Reads an image from a local file.
        """
        
        return cv2.imread(str(image_filename), imread_flag)

    def read_image_from_url(image_url, imread_flag):
        """
        Reads an image from a URL.
        """
        url_response = urlopen(str(image_url))
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, imread_flag)
    
    imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    local_file = os.path.exists(image_uri) 

    try:
        img = None
        if local_file:
            img = read_image_from_filename(image_uri, imread_flag)
        else:
            img = read_image_from_url(image_uri, imread_flag)

        assert img is not None
    
    except Exception as e:
        raise ValueError(f"Error reading image from {image_uri}: {e}")
    
    return img

def write_image(image: np.ndarray, filename: Union[Path, str]) -> None:
    """
    Writes an image to a file.
    """
    cv2.imwrite(str(filename), image)

def compute_sha256(file_path: Union[Path, str]) -> str:
    """
    Computes the SHA256 hash of a file.
    """
    with open(file_path, 'rb') as f:
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

    