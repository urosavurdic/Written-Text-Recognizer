# Utility functions for text recognition

import numpy as np

def to_categorical(y, num_classes):
    """
    Convert a 1D array of class indices to a 2D one-hot encoded array.
    
    Args:
        y (np.ndarray): 1D array of class indices.
        num_classes (int): Total number of classes.
        
    Returns:
        np.ndarray: 2D one-hot encoded array.
    """
    return np.eye(num_classes)[y] if y.ndim == 1 else np.eye(num_classes)[y.argmax(axis=1)]

