from typing import Any, Dict, Callable, Sequence, Union, Tuple
import torch

SeqOrTensor = Union[Sequence, torch.Tensor]

class BaseDataset(torch.utils.data.Dataset):

    """
    Base class for datasets that simply provides a structure for data and target. 
    It can be extended to include additional functionality such as transformations.
    Args:
        data (SeqOrTensor): The input data, which can be a sequence or a tensor.
        target (SeqOrTensor): The target data, which can also be a sequence or a tensor.
        transform (Callable, optional): A function/transform that takes in an input sample and returns a transformed version.
        target_transform (Callable, optional): A function/transform that takes in the target and returns a transformed version
    """
    
    def __init__(self, data: SeqOrTensor, target: SeqOrTensor, transform: Callable = None, target_transform: Callable = None) -> None:
        if len(data) != len(target):
            raise ValueError("Data and targets must be of equal length") # raise an error if lengths do not match - common mistake
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, index: int):
        """
        Returns a single item from the dataset.
        Args:
            index (int): Index of the item to retrieve.
        Returns:
            Tuple[SeqOrTensor, SeqOrTensor]: A tuple containing the data and target at the specified index.
        """
        data_item = self.data[index]
        target_item = self.target[index]
        
        # Apply transformations if they are provided

        if self.transform is not None:
            data_item = self.transform(data_item)
        
        if self.target_transform is not None:
            target_item = self.target_transform(target_item)
        
        return data_item, target_item
    

def convert_str_to_labels(text: str, char_to_idx: Dict[str, int], max_length: int) -> torch.Tensor:
    """
    Converts a string to a tensor of labels based on a character-to-index mapping.
    The mapping should include special tokens for:
        - Start of sequence (`<S>`)
        - End of sequence (`<E>`)
        - Padding (`<P>`)
    
    Args:
        text (str): The input string to convert.
        char_to_idx (Dict[str, int]): A dictionary mapping characters to their corresponding indices, including special tokens.
        max_length (int, optional): The maximum length of the output tensor. If provided, the output will be padded or truncated to this length.

    """
    tokens = ["<S>"] + list(text) + ["<E>"]  # Add start and end tokens
    labels = torch.ones((max_length), dtype=torch.long) * char_to_idx['<P>'] # Padding character
    for i, token in enumerate(tokens[:max_length]): # 
        labels[i] = char_to_idx[token]
    
    return labels

def split_dataset(base_dataset: BaseDataset, split_ratio: float = 0.8, seed: int = 42) -> Tuple[BaseDataset, BaseDataset]:
    """
    Splits a dataset into two subsets based on a specified ratio.
    
    Args:
        base_dataset (BaseDataset): The dataset to split.
        split_ratio (float): The ratio of the first subset size to the total dataset size.
        seed (int): Random seed for reproducibility.
    
    Returns:
        Tuple[BaseDataset, BaseDataset]: Two datasets resulting from the split.
    """
    split_1_size = int(len(base_dataset) * split_ratio)
    split_2_size = len(base_dataset) - split_1_size

    return torch.utils.data.random_split(base_dataset, [split_1_size, split_2_size], generator=torch.Generator().manual_seed(seed))

"""
#Testing the BaseDataset class and utility
xs = [torch.randn(3, 4) for _ in range(10)]
ys = list(range(10))
ds = BaseDataset(xs, ys)
assert len(ds) == 10 
assert ds[0][1] == 0

def double(x): return x * 2 
ds = BaseDataset(xs, ys, transform=double)
assert torch.allclose(ds[0][0], xs[0] * 2) # check if transform is applied correctly

try:
    BaseDataset([1,2,3], [1,2])
except ValueError:
    pass
else:
    raise AssertionError("Should raise ValueError")

print(ds[0])
print(split_dataset(ds, 0.8))
print(convert_str_to_labels("hi", {"<P>":0,"<S>":1,"<E>":2,"h":3,"i":4}, 6))
"""