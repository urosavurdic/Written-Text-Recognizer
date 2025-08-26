import argparse
import torch
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy as TorchAccuracy


OPTIMIZER = 'Adam'
LR = 1e-3
LOSS = 'cross_entropy'
ONE_CYCLE_TOTAL_STEPS = 100

class Accuracy(TorchAccuracy):
    """Accuracy Metric with a hack."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Hack for PyTorch Lightning 1.2+ softmax issue.
        """
        if preds.min() < 0 or preds.max() > 1:
            preds = F.softmax(preds, dim=-1)
        super().update(preds=preds, target=target)
class BaseModel(pl.LightningModule):
    """
    Base class for all models in the text recognizer project. It provides a common interface for model initialization:
    - `model`: The neural network model to be trained.
    - `args`: Optional arguments for model configuration, such as optimizer type and learning rate.
    - `optimizer`: The optimizer to be used for training, defaulting to Adam.
    - `lr`: Learning rate for the optimizer, defaulting to 1e-3.
    - `loss_fn`: The loss function to be used, defaulting to cross-entropy loss.
    - `train_acc`, `val_acc`, `test_acc`: Accuracy metrics for training, validation, and testing phases.
    This class is designed to be extended by specific model implementations, such as MLP or CNN, which will define their own architectures and training logic.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get('optimizer', OPTIMIZER)
        self.optimizer = getattr(torch.optim, optimizer)
        self.lr = self.args.get('lr', LR)

        loss = self.args.get('loss', LOSS)
        if loss not in ("ctc", "transformer"):
            self.loss_fn = getattr(torch.nn.functional, loss)

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
    
    def configure_optimizers(self):
        """
        Configures the optimizer for the model based on the specified optimizer type and learning rate.
        Returns:
            torch.optim.Optimizer: The configured optimizer instance.
        """
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def forward(self, x):
        """
        Forward pass of the model. This method should be overridden by subclasses to define the model's architecture.
        Args:
            x (torch.Tensor): Input tensor to the model.
        Returns:
            torch.Tensor: Output tensor from the model.
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Training step for the model. This method should be overridden by subclasses to define the training logic.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Loss value for the current training step.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model. This method should be overridden by subclasses to define the validation logic.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the current batch.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss)
        self.val_acc(logits, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        """
        Test step for the model. This method should be overridden by subclasses to define the testing logic.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the current batch.
        """
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    @staticmethod
    def add_to_argparse(parser):
        """
        Adds model-specific arguments to the provided argument parser.
        Args:
            parser (argparse.ArgumentParser): The argument parser to which model-specific arguments will be added.
        Returns:
            argparse.ArgumentParser: The updated argument parser with model-specific arguments.
        """
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser
