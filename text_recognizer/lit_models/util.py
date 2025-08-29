from typing import Union
import torch

def first_element(x: torch.Tensor, element: Union[int, float], dim: int = 1) -> torch.Tensor:
    """
    This function efficiently computes the index of the first occurrence of a value along a dimension using cumulative sums and boolean masks.
    If the element doesn’t exist, it returns the length of the dimension (a “sentinel” value).
    example:
        x = torch.tensor([[1, 2, 3],
                  [2, 3, 3],
                  [1, 1, 1]])

        first_element(x, 3)
        explanation: 
            Row 1 → first 3 at index 2
            Row 2 → first 3 at index 1
            Row 3 → no 3 → return 3
        return:
            tensor([2, 1, 3])
    Based on https://discuss.pytorch.org/t/first-nonzero-index/24769/9
    """
    nonz = x == element
    ind = ((nonz.cumsum(dim) == 1) & nonz).max(dim).indices
    ind[ind == 0] = x.shape[dim]
    
    return ind