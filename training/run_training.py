def main():
    """
    Main function to run the training script.
    """
    parser = _setup_parser()
    args = parser.parse_args()

    # Import the data and model classes based on the arguments
    data_class = _import_class(f'text_recognizer.data.{args.data_class}')
    model_class = _import_class(f'text_recognizer.models.{args.model_class}')
    
    # Instantiate the dataset
    data = data_class(args)
    
    # Instantiate the model
    model = model_class(data_config=data.configuration(), args=args)
    
    # Choose the appropriate LitModel class (do not instantiate yet)
    if args.loss not in ("ctc", "transformer"):
        lit_model_class = lit_models.BaseModel
    elif args.loss == "ctc":
        lit_model_class = lit_models.CTCLitModel
    elif args.loss == "transformer":
        lit_model_class = lit_models.TransformerLitModel

    # Instantiate the LitModel
    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(
            args.load_checkpoint, args=args, model=model, num_classes=data.num_classes
        )
    else:
        lit_model = lit_model_class(model=model, args=args, num_classes=data.num_classes)
        
    # Set up logger and callbacks
    logger = [pl.loggers.TensorBoardLogger("training/logs")]

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=10
    )
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", 
        monitor="val_loss", mode="min"
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]

    args.weight_summary = 'full'  # print full model summary
    
    # Initialize trainer
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, logger=logger, weights_save_path="training/logs"
    )

    # Tune learning rate (optional)
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(lit_model, datamodule=data)
    # lit_model.hparams.lr = lr_finder.suggestion()

    # Train and test
    trainer.tune(lit_model, datamodule=data)
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
