import argparse
import importlib

import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
#from pytorch_lightning.tuner import Tuner
import wandb


from text_recognizer import lit_models

np.random.seed(42)
torch.manual_seed(42)

def _import_class(module_and_class_name: str) -> type:
    """
    Import a class from a module given its full path.
    Args:
        module_and_class_name (str): Full path to the class, e.g., 'text_recognizer.models.MLPModel'.
    """
    module_name, class_name = module_and_class_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def _setup_parser():
    """
    Set up the argument parser for the training script.
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(add_help=False)
    
    # Add basic trainer arguments manually
    
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args" 
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])
    """
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--accelerator', type=str, default="cpu")   # use "gpu" if available
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', type=int, default=32)
    """
    
    #parser.add_argument("--wandb", action="store_true", default=False)

    # basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument('--data_class', type=str, default='EMNIST')
    parser.add_argument('--model_class', type=str, default='CNN')
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # model specific arguments
    temp_arg, _ = parser.parse_known_args()
    model_class = _import_class(f'text_recognizer.models.{temp_arg.model_class}')
    data_class = _import_class(f'text_recognizer.data.{temp_arg.data_class}')

    # get data, model and LitModel specific arguments
    data_group = parser.add_argument_group('Data Arguments')
    data_class.add_arguments(data_group)

    model_group = parser.add_argument_group('Model Arguments')
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group('LitModel Arguments')
    lit_models.BaseModel.add_to_argparse(lit_model_group)

    return parser

def main():
    """
    Main function to run the training script.
    python run_training.py --max_epochs 3 --gpus='0' --num_workers 4 --model_class=MLP --data_class=MNIST
    """
    parser = _setup_parser()
    args = parser.parse_args()

    #Import the data and model classes based on the arguments
    data_class = _import_class(f'text_recognizer.data.{args.data_class}')
    model_class = _import_class(f'text_recognizer.models.{args.model_class}')
    
    data = data_class(args)
    #data.setup("fit") 
    #print("Num classes:", data.num_classes)
    #print("Max label in dataset:", max([int(y) for y in data.targets]))
    model = model_class(data_config=data.configuration(), args=args)
    # choosing right model
    if args.loss not in ("ctc", "transformer"):
        lit_model_class = lit_models.BaseModel
    if args.loss == "ctc":
        lit_model_class = lit_models.CTCModel
    if args.loss == "transformer":
        lit_model_class = lit_models.TransformerModel
    
    # num_classes = getattr(data, "num_classes", None) only for EMNIST MNIST and IAM

    # load from checkpoint
    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(model=model, args=args)
        
    logger = pl.loggers.TensorBoardLogger("training/logs")
    
    if args.wandb:
        logger = pl.loggers.WandbLogger()
        logger.watch(model)
        logger.log_hyperparams(vars(args))

    

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]

    args.weight_summary = 'full' # print full model summary
    trainer_kwargs = {
        #"wandb": args.wandb,
        "max_epochs": args.max_epochs,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "precision": args.precision,
        "logger": logger,
        "callbacks": callbacks,
        "enable_checkpointing": True,  # to ensure ModelCheckpoint works
    }

    trainer = pl.Trainer(**trainer_kwargs)
    #trainer = pl.Trainer(**trainer_args, callbacks=callbacks, logger=logger, weights_save_path="training/logs")
    
    trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
    
    # Save the best model path
    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        print("Best model saved at:", best_model_path)
        if args.wandb:
            wandb.save(best_model_path)
            print("Best model also uploaded to W&B")

    
if __name__ == '__main__':
    main()

