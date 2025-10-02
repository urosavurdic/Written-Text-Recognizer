from typing import Sequence
import pytorch_lightning as pl
import torch
import editdistance
import torchmetrics

class CharacterErrorRate(torchmetrics.Metric):
    """
    This class implements Character Error Rate (CER) — a metric for evaluating sequence models (e.g. OCR, speech recognition).
    It compares predicted sequences (preds) to target sequences (targets) using Levenshtein (edit) distance.
    CER = edit distance / max sequence length
    """
    def __init__(self, ignore_tokens: Sequence[int], *args):
        super().__init__(*args)
        self.ignore_tokens = set(ignore_tokens) # tokens to skip when evaluating
        # registers internal states for metrics: 
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum") # accumulated CER errors (float)
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum") # total: number of samples processed (int)
        # dist_reduce_fx="sum": ensures values are summed correctly when running on multiple GPUs
        self.error: torch.Tensor
        self.total: torch.Tensor

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        N = preds.shape[0]
        for ind in range(N): # for each sample in the batch
            # convert predictions and targets to Python lists and remove ignore_tokens
            pred = [_ for _ in preds[ind].tolist() if _ not in self.ignore_tokens]
            target = [_ for _ in targets[ind].tolist() if _ not in self.ignore_tokens]
            """
            Compute Levenshtein distance using editdistance package.
            * Levenshtein distance = minimum number of insertions, deletions, or substitutions to transform one sequence into the other
            """
            distance = editdistance.distance(pred, target)
            # normalize by sequence length → per-sample CER.
            error = distance / max(len(pred), len(target))
            # accumulate into self.error
            self.error = self.error + error
        
        # increment self.total by the batch size
        self.total = self.total + N

    
    def compute(self) -> torch.Tensor:
        # Returns the average CER across all processed samples
        return self.error / self.total
    
"""
Example usage and test case for CharacterErrorRate metric.

def test_character_error_rate():
    metric = CharacterErrorRate([0, 1])
    X = torch.tensor(
        [
            [0, 2, 2, 3, 3, 1],  # identical to Y[0] => error will be 0
            [0, 2, 1, 1, 1, 1],  # very different from Y[1] => error will be .75
            [0, 2, 2, 4, 4, 1],  # partially different from Y[2] => error will be .5
        ]
    )
    Y = torch.tensor(
        [
            [0, 2, 2, 3, 3, 1],
            [0, 2, 2, 3, 3, 1],
            [0, 2, 2, 3, 3, 1],
        ]
    )
    metric(X, Y)
    print(metric.compute())
    assert metric.compute() == sum([0, 0.75, 0.5]) / 3

    if __name__ == "__main__":
        test_character_error_rate()
"""  
