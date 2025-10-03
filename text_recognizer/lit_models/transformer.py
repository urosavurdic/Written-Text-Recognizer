import torch.nn as nn
import wandb
from .metrics import CharacterErrorRate
from .base import BaseModel

class TransformerModel(BaseModel):
    def __init__(self, model, args = None):
        super().__init__(model, args)
        self.char_to_idx = self.model.data_config["char_to_idx"]
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        start_index = self.char_to_idx["<S>"]
        self.blank_index = self.char_to_idx["<B>"]
        end_index = self.char_to_idx["<E>"]
        padding_index = self.char_to_idx["<P>"]

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=padding_index)

        ignore_tokens = [start_index, end_index, padding_index]
        self.val_cer = CharacterErrorRate(ignore_tokens)
        self.test_cer = CharacterErrorRate(ignore_tokens)

    def forward(self, x):
        return self.model.predict(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x, y[:, :-1])
        loss = self.loss_fn(logits, y[:, 1:])
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x, y[:, :-1])
        loss = self.loss_fn(logits, y[:, 1:])
        self.log("val_loss", loss, prog_bar=True)

        pred = self.model.predict(x)

        pred_str = "".join(self.idx_to_char[_] for _ in pred[0].tolist() if _ != 3)
        try:
            self.logger.experiment.log({"val_pred_examples": [wandb.Image(x[0], caption=pred_str)]})
        except AttributeError:
            pass
        
        self.val_cer(pred, y)
        self.log("val_cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model.predict(x)

        pred_str = "".join(self.idx_to_char[_] for _ in pred[0].tolist() if _ != 3)
        try:
            self.logger.experiment.log({"test_pred_examples": [wandb.Image(x[0], caption=pred_str)]})
        except AttributeError:
            pass

        self.test_cer(pred, y)
        self.log("test_cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)