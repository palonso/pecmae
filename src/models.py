from collections import defaultdict

import pytorch_lightning as L
import numpy as np
import torchmetrics
from torch import optim, nn
from sklearn.metrics import classification_report


class MLP(L.LightningModule):
    def __init__(
        self,
        in_size: int = 2096,
        out_size: int = 10,
        hidden_size: int = 128,
        lr: float = 1e-4,
        labels=None,
    ):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )
        self.loss = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=out_size,
        )
        self.out_size = out_size

        self.aggregated_y_hat_test = defaultdict(list)
        self.filename_to_label = defaultdict(list)
        self.lr = lr
        self.labels = labels

    def training_step(self, batch, batch_idx, split="train"):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["feature"]
        y = batch["label"]
        f = batch["filename"]

        x = x.squeeze()

        y_hat = self.MLP(x)
        loss = self.loss(y_hat, y)
        self.log(
            f"{split}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if split == "test":
            for f_sample, y_hat_sample, y_sample in zip(f, y_hat, y):
                self.aggregated_y_hat_test[f_sample].append(
                    y_hat_sample.detach().cpu().numpy()
                )
                if f_sample not in self.filename_to_label:
                    self.filename_to_label[f_sample] = y_sample.detach().cpu().numpy()

        acc = self.accuracy(y_hat, y)

        self.log(
            f"{split}_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx, split="test")

    def on_test_end(self):
        y = []
        y_hat = []
        for f, y_hat_aggregated in self.aggregated_y_hat_test.items():
            y_hat_aggregated = np.array(y_hat_aggregated)
            y_hat.append(np.argmax(np.average(y_hat_aggregated, axis=0), axis=0))
            y.append(self.filename_to_label[f])

        y_hat = np.array(y_hat)
        y = np.array(y)

        print(classification_report(y, y_hat, target_names=self.labels, digits=3))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer
