import argparse
import torch
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback



OPTIMIZER = 'Adam'
LR = 1e-3
LOSS = 'cross_entropy'
ONE_CYCLE_TOTAL_STEPS = 100

def my_accuracy(logits, y):
    """
    Newer versions of accuracy require num_classes. As previous trial to implement num_classes faced many issues, I implement simple accuracy calculation here.
    """
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean()



class BaseModel(pl.LightningModule):
    """
    Base class for all models in the text recognizer project. It provides a common interface for model initialization:
    - `model`: The neural network model to be trained.
    - `args`: Optional arguments for model configuration, such as optimizer type and learning rate.
    - `optimizer`: The optimizer to be used for training, defaulting to Adam.
    - `lr`: Learning rate for the optimizer, defaulting to 1e-3.
    - `loss_fn`: The loss function to be used, defaulting to cross-entropy loss.
    - `train_acc`, `val_acc`, `test_acc`: Accuracy metrics for training, validation, and testing phases for each batch.
    This class is designed to be extended by specific model implementations, such as MLP or CNN, which will define their own architectures and training logic.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}
        """
        self.num_classes = len(self.args.get('char_to_idx', data_config['char_to_idx']))
        """


        optimizer = self.args.get('optimizer', OPTIMIZER)
        self.optimizer = getattr(torch.optim, optimizer)
        self.lr = self.args.get('lr', LR)

        loss = self.args.get('loss', LOSS)
        if loss not in ("ctc", "transformer"):
            self.loss_fn = getattr(torch.nn.functional, loss)
        
        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        self.train_accs = []
        self.val_accs = []
        self.test_accs = []
        
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

        logits = self(x)
        loss = self.loss_fn(logits, y)

        batch_acc = my_accuracy(logits, y)
        self.train_accs.append(batch_acc)

        self.log('train_loss', loss)
        
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
        acc = my_accuracy(logits, y)
        self.val_accs.append(acc)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
   
    
    def test_step(self, batch, batch_idx):
        """
        Test step for the model. This method should be overridden by subclasses to define the testing logic.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the current batch.
        """
        x, y = batch
        logits = self(x)
        
        acc = my_accuracy(logits, y)

        self.test_accs.append(acc)

    # https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
    
    def on_train_epoch_end(self):
        if self.train_accs:
            epoch_acc = torch.stack(self.train_accs).mean()
            self.log('train_acc', epoch_acc, prog_bar=True, logger=True)
            # Clear the list after logging to prevent accumulation
            self.train_accs = []
    
    def on_validation_epoch_end(self):
        if self.val_accs:
            epoch_acc = torch.stack(self.val_accs).mean()
            self.log('val_acc', epoch_acc, prog_bar=True, logger=True)
            # Clear the list after logging
            self.val_accs = []
    
    def on_test_epoch_end(self):
        if self.test_accs:
            epoch_acc = torch.stack(self.test_accs).mean()
            self.log('test_acc', epoch_acc, prog_bar=True, logger=True)
            # Clear the list after logging
            self.test_accs = []
    
    def on_train_epoch_start(self):
        # Reset list at the start of each epoch as backup
        self.train_accs = []
    
    def on_validation_epoch_start(self):
        # Reset list at the start of each epoch as backup
        self.val_accs = []
    
    def on_test_epoch_start(self):
        # Reset list at the start of each epoch as backup
        self.test_accs = []

    
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
