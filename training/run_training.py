import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
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

    # add trainer specific arguments --max_epochs, --gpus, --precision, etc.
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Arguments"
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])
    
    # basic arguments
    parser.add_argument('--data_class', type=str, default='MNIST')
    parser.add_argument('--model_class', type=str, default='MLP')

    # model specific arguments
    temp_arg = parser.parse_known_args()
    model_class = _import_class(f'text_recognizer.models.{temp_arg.data_class}')
    data_class = _import_class(f'text_recognizer.data.{temp_arg.model_class}')

    # get data, model and LitModel specific arguments
    data_group = parser.add_argument_group('Data Arguments')
    data_class.add_data_specific_args(data_group)

    model_group = parser.add_argument_group('Model Arguments')
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group('LitModel Arguments')
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument('--help', '-h', action='help')
    return parser

def main():
    """
    Main function to run the training script.
    """
    parser = _setup_parser()
    args = parser.parse_args()

    #Import the data and model classes based on the arguments
    data_class = _import_class(f'text_recognizer.data.{args.data_class}')
    model_class = _import_class(f'text_recognizer.models.{args.model_class}')
    
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)

    lit_model = lit_models.BaseLitModel(model, args=args)

    loggers = [pl.loggers.TensorBoardLogger("training/logs")]

    callbacks = [pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)]

    args.weight_summary = 'full' # print full model summary
    trainer = pl.Trainer.from_argparse_args(args, logger=loggers, callbacks=callbacks)

    trainer.tune(lit_model, datamodule=data) # if passing --auto_lr_find, this will find the optimal learning rate
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)

if __name__ == '__main__':
    main()

