import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from bert_base import BERT


class BertClassifier(pl.LightningModule):
    def __init__(self, n_classes, d_model=768, n_layers=12, n_heads=12):
        super().__init__()
        self.save_hyperparameters()
        self.n_classes = n_classes
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.encode = BERT(
            self.tokenizer.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        self.output = nn.Linear(d_model, n_classes)

        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = list(x)  # tuple -> list
        y = y.float()

        o = self(x)
        loss = F.binary_cross_entropy(o, y)

        self.train_acc(o, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log(
            "train_acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = list(x)  # tuple -> list
        y = y.float()

        o = self(x)
        loss = F.binary_cross_entropy(o, y)

        self.val_acc(o, y)
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=False, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-05)

    def forward(self, x):
        if type(x) == "str":
            x = [x]
        x = self.tokenizer(x, return_tensors="pt", padding=True)["input_ids"]
        if self.device.type == "cuda":
            x = x.cuda()
        x = self.encode(x)
        # x = x.last_hidden_state
        x = x[:, 0]  # take only [CLS] token
        x = self.output(x)
        x = F.softmax(x, dim=-1) if self.n_classes > 2 else torch.sigmoid(x)
        return x[:, 0]
