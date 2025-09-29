import argparse
import torch
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torchmetrics.functional import accuracy


OPTIMIZER = 'Adam'
LR = 1e-3
LOSS = 'cross_entropy'
ONE_CYCLE_TOTAL_STEPS = 100


class BaseModel(pl.LightningModule, Callback):
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

        self.processed_train_samples = 0
        self.processed_val_samples = 0
        self.processed_test_samples = 0
        
        self.train_acc = 0
        self.val_acc = 0
        self.test_acc = 0
        
        #self.num_classes = len(self.char_to_idx)
        """
        if num_classes is None:
            raise ValueError("num_classes must be provided to BaseModel")
        self.num_classes = num_classes
        

        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        """
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
        size = x.size(0)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)

        batch_acc = accuracy(logits, y)
        self.train_acc = ((self.processed_train_samples * self.train_acc + size * batch_acc) / (self.processed_train_samples + size))
        self.processed_train_samples += size

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
        
        size = x.size(0)
        batch_acc = accuracy(logits, y)
        self.val_acc = ((self.processed_val_samples * self.val_acc + size * batch_acc) / (self.processed_val_samples + size))
        self.processed_val_samples += size

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
        size = x.size(0)
        batch_acc = accuracy(logits, y)
        self.test_acc = ((self.processed_test_samples * self.test_acc + size * batch_acc) / (self.processed_test_samples + size))

        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    # https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
    
    def on_train_epoch_start(self) -> None:
        self.train_acc = 0
        self.processed_train_samples = 0

    def on_validation_epoch_start(self) -> None:
        self.val_acc = 0
        self.processed_val_samples = 0

    def on_test_epoch_start(self) -> None:
        self.test_acc = 0
        self.processed_test_samples = 0

    
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
