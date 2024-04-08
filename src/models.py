from collections import defaultdict
from pathlib import Path
from typing import Any

import pytorch_lightning as L
import numpy as np
import torch
import torchmetrics
from torch import optim, nn
from sklearn.metrics import classification_report

from similarities import InfoNCE


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


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        global lambd
        return grad_output * -lambd


def grad_reverse(x):
    return GradReverse.apply(x)


class domain_classifier_linear(nn.Module):
    def __init__(self, input_dim=1200):
        super(domain_classifier_linear, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.fc1(x)
        return x


class domain_classifier_mlp(nn.Module):
    def __init__(self, input_dim=1200):
        super(domain_classifier_mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(32, 1)
        self.drop = nn.Dropout1d(0.2)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return x


class domain_classifier_conv(nn.Module):
    def __init__(self):
        super(domain_classifier_conv, self).__init__()

        self.cv1 = nn.Conv2d(1, 16, 5)
        self.mp1 = nn.MaxPool2d(2)
        self.cv2 = nn.Conv2d(16, 16, 5)
        self.mp2 = nn.MaxPool2d(2)
        self.cv3 = nn.Conv2d(16, 32, 5)
        self.mp3 = nn.MaxPool2d(4)
        self.cv4 = nn.Conv2d(32, 32, 5)
        self.mp4 = nn.MaxPool2d(4)
        self.ln1 = nn.Linear(960, 1)

        self.leaky_relu = nn.LeakyReLU()
        self.drop = nn.Dropout1d(0.25)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = grad_reverse(x)

        x = self.leaky_relu(self.mp1(self.cv1(x)))
        x = self.leaky_relu(self.mp2(self.cv2(x)))
        x = self.leaky_relu(self.mp3(self.cv3(x)))
        x = self.leaky_relu(self.mp4(self.cv4(x)))
        x = x.flatten(1)

        x = self.leaky_relu(self.drop(self.ln1(x)))

        return x


class DenseRes(nn.Module):
    def __init__(self, feat_dim, dropout=0.2):
        super(DenseRes, self).__init__()

        self.feat_dim = feat_dim
        self.dropout = dropout

        self.bn1 = nn.BatchNorm1d(self.feat_dim)
        self.ln1 = nn.Linear(self.feat_dim, self.feat_dim)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(self.feat_dim, self.feat_dim)
        self.ln2 = nn.Linear(self.feat_dim, self.feat_dim)

        self.drop = nn.Dropout1d(self.dropout)

    def forward(self, x):
        x = x.flatten(1)

        y = self.relu(self.ln1(x))
        x = self.bn1(x + y)
        x = self.drop(x)
        y = self.relu(self.ln2(x))
        x = self.bn2(x + y)

        x.unsqueeze(1)

        return x


class PrototypeNet(L.LightningModule):
    def __init__(
        self,
        protos: np.ndarray,
        time_dim: int = 300,
        feat_dim: int = 768,
        n_labels: int = 10,
        batch_size: int = 32,
        total_steps: int = 1000,
        weight_decay: float = 1e-4,
        max_lr: float = 1e-3,
        temp: float = 0.1,
        alpha: float = 0.5,
        proto_loss: str = "l2",
        proto_loss_samples: str = "any_sample",
        use_discriminator: bool = False,
        discriminator_type: str = "mlp",
        distance: str = "l2",
        freeze_protos: bool = False,
        time_summarization: str = "none",
        do_normalization: bool = False,
        ds_mean: Any = None,
        ds_std: Any = None,
        labels: Any = None,
    ):
        super().__init__()

        self.time_dim = time_dim
        self.feat_dim = feat_dim
        self.n_labels = n_labels
        self.protos_weights = protos
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.weight_decay = weight_decay
        self.max_lr = max_lr
        self.temp = temp
        self.alpha = alpha
        self.proto_loss = proto_loss
        self.proto_loss_samples = proto_loss_samples
        self.use_discriminator = use_discriminator
        self.discriminator_type = discriminator_type
        self.distance = distance
        self.save_protos_each_n_steps = 50000
        self.time_summarization = time_summarization
        self.do_normalization = do_normalization
        self.ds_mean = ds_mean
        self.ds_std = ds_std
        self.labels = labels

        self.n_protos = self.protos_weights.shape[0]
        self.n_protos_per_label = self.n_protos // self.n_labels

        self.protos = nn.Parameter(
            data=torch.tensor(self.protos_weights), requires_grad=not freeze_protos
        )
        self.linear = nn.Linear(self.n_protos, self.n_labels)

        # init linear weights to direct connection to the class
        lin_weights = np.hstack(
            [[i] * self.n_protos_per_label for i in range(self.n_labels)]
        )
        lin_weights = torch.nn.functional.one_hot(torch.tensor(lin_weights))
        self.linear.weight = nn.Parameter(
            data=lin_weights.T.float(), requires_grad=True
        )

        self.info_nce = InfoNCE(negative_mode=None)
        self.xent = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=n_labels,
        )

        if self.use_discriminator:
            if self.discriminator_type == "conv":
                self.discriminator = domain_classifier_conv()

            elif self.discriminator_type == "mlp":
                self.discriminator = domain_classifier_mlp(
                    input_dim=self.time_dim * self.feat_dim
                )
            elif self.discriminator_type == "linear":
                self.discriminator = domain_classifier_linear(
                    input_dim=self.time_dim * self.feat_dim
                )
            else:
                raise ValueError(
                    f"discriminator type {self.discriminator_type} not supported"
                )

            self.bxent = nn.BCEWithLogitsLoss()
            self.accuracy_binary = torchmetrics.classification.Accuracy(task="binary")

        # to use the scheduler
        self.automatic_optimization = False

        self.i = 0

        if self.time_summarization == "lstm":
            self.time_summarizer = nn.LSTM(
                input_size=self.feat_dim,
                hidden_size=self.feat_dim,
                num_layers=2,
                dropout=0.2,
                batch_first=True,
            )

        if self.time_summarization == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.feat_dim,
                nhead=1,
                batch_first=True,
            )
            self.time_summarizer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=1,
            )
        if self.time_summarization == "dense_res":
            self.time_summarizer = DenseRes(self.feat_dim)

        self.aggregated_y_hat = defaultdict(list)
        self.filename_to_label = defaultdict(list)

    def training_step(self, batch, batch_idx, split="train"):
        x = batch["feature"]
        y = batch["label"]
        f = batch["filename"]

        # flatten time and embedding dimension
        if self.time_summarization == "none":
            x = x.flatten(1)
            protos = self.protos.flatten(1)

        elif self.time_summarization == "lstm":
            protos = self.time_summarizer(self.protos)[0].flatten(1)
            x = x.flatten(1)
        elif self.time_summarization in ("transformer", "dense_res"):
            protos = self.time_summarizer(self.protos).flatten(1)
            x = x.flatten(1)
        else:
            raise ValueError(
                f"time summarization {self.time_summarization} not supported"
            )

        optimizers = self.optimizers()
        lr_schedulers = self.lr_schedulers()

        optimizer_c = optimizers
        lr_scheduler_c = lr_schedulers

        if split == "train":
            optimizer_c.zero_grad()

        if self.proto_loss == "info_nce":
            similarity = self.info_nce(x, protos, output="logits")
            distance = torch.exp(-similarity)
        elif self.proto_loss == "l1":
            distance = torch.mean(torch.abs((x.unsqueeze(1) - protos.unsqueeze(0))), -1)
            similarity = torch.exp(-distance)
        elif self.proto_loss == "l2":
            distance = torch.mean((x.unsqueeze(1) - protos.unsqueeze(0)) ** 2, -1)
            similarity = torch.exp(-distance)
        else:
            raise ValueError(f"distance {self.proto_loss} not supported")

        self.log(f"{split}_distance", distance.mean(), batch_size=self.batch_size)

        y_hat = self.linear(similarity)
        acc = self.accuracy(y_hat, y)

        if split in ("val", "test"):
            for f_sample, y_hat_sample, y_sample in zip(f, y_hat, y):
                self.aggregated_y_hat[f_sample].append(
                    y_hat_sample.detach().cpu().numpy()
                )
                if f_sample not in self.filename_to_label:
                    self.filename_to_label[f_sample] = y_sample.detach().cpu().numpy()

        # classification loss
        loss_c = self.xent(y_hat / self.temp, y)
        # prototype loss

        if self.proto_loss_samples == "all":
            loss_p = torch.mean(torch.min(distance, dim=0).values)

        elif self.proto_loss_samples == "class":
            distance_mask = torch.inf * torch.ones_like(distance)
            idx = torch.zeros_like(distance_mask)

            for j in range(self.n_protos_per_label):
                idx += torch.nn.functional.one_hot(
                    y * self.n_protos_per_label + j,
                    num_classes=distance.shape[1],
                )

            # Index with a boolean mask (similar to numpy).
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#boolean-array-indexing
            idx = idx.bool()

            distance_mask[idx] = distance[idx]

            min_dis = torch.min(distance_mask, dim=0).values
            min_dis_clean = min_dis[torch.where(min_dis != torch.inf)]

            loss_p = torch.mean(min_dis_clean)

        loss = self.alpha * loss_p + (1 - self.alpha) * loss_c

        self.log(f"{split}_acc", acc, prog_bar=True, batch_size=self.batch_size)
        self.log(f"{split}_class_loss", loss_c, batch_size=self.batch_size)
        self.log(f"{split}_proto_loss", loss_p, batch_size=self.batch_size)

        if self.use_discriminator:
            disc_weight = 1.0

            p = float(self.i) / self.total_steps
            global lambd

            lambd = disc_weight * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1)

            x_disc = torch.cat([x, protos], dim=0)
            y_disc = torch.cat(
                [
                    torch.zeros(len(y), device=self.device),
                    torch.ones(self.n_protos, device=self.device),
                ],
                dim=0,
            )
            y_disc = y_disc.unsqueeze(1)

            if self.discriminator_type == "conv":
                y_disc_hat = self.discriminator(
                    x_disc.reshape(-1, self.time_dim, self.feat_dim)
                )
            else:
                y_disc_hat = self.discriminator(x_disc)

            loss_d = self.bxent(y_disc_hat, y_disc)
            acc_disc = self.accuracy_binary(y_disc_hat, y_disc)

            loss += loss_d

            self.log(f"{split}_disc_acc", acc_disc, batch_size=self.batch_size)

            if split == "train":
                self.log("lambda", lambd, batch_size=self.batch_size)

            self.log(f"{split}_disc_loss", loss_d, batch_size=self.batch_size)

        if split == "train":
            self.manual_backward(loss)
            optimizer_c.step()

            lr_scheduler_c.step()
            self.log(
                f"{split}_lr_class",
                lr_scheduler_c.get_last_lr()[0],
                batch_size=self.batch_size,
            )

            if self.i % self.save_protos_each_n_steps == 0 and self.i != 0:
                self.save_checkpoint()

        self.i += 1

    def validation_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        optimizer_c = optim.Adam(
            self.parameters(),
            weight_decay=self.weight_decay,
        )
        scheduler_c = optim.lr_scheduler.OneCycleLR(
            optimizer_c,
            max_lr=self.max_lr,
            total_steps=self.total_steps,
        )
        optimizers = [
            {"optimizer": optimizer_c, "lr_scheduler": scheduler_c},
        ]

        return optimizers

    def save_checkpoint(self):
        if self.time_summarization != "none":
            with torch.no_grad():
                protos = self.time_summarizer(self.protos)

            if self.time_summarization == "lstm":
                protos = protos[0]
        else:
            protos = self.protos

        protos = protos.detach().cpu().numpy()

        if self.do_normalization:
            protos = (protos * self.ds_std * 2) + self.ds_mean

        out_data_dir = Path("out_data/")
        out_data_dir.mkdir(exist_ok=True)

        protos_file = out_data_dir / f"protos_v{self.logger.version}_s{self.i}.npy"
        print(f"saving protos to {protos_file}")

        np.save(protos_file, protos)

    def on_validation_epoch_end(self):
        y = []
        y_hat = []
        for f, y_hat_aggregated in self.aggregated_y_hat.items():
            y_hat_aggregated = np.array(y_hat_aggregated)
            y_hat.append(np.argmax(np.average(y_hat_aggregated, axis=0), axis=0))
            y.append(self.filename_to_label[f])

        y_hat = np.array(y_hat)
        y = np.array(y)

        acc = self.accuracy(torch.Tensor(y_hat), torch.Tensor(y))
        self.log(f"val_acc_aggregated", acc, batch_size=self.batch_size)

        self.aggregated_y_hat = defaultdict(list)

    def on_test_end(self):
        y = []
        y_hat = []
        for f, y_hat_aggregated in self.aggregated_y_hat.items():
            y_hat_aggregated = np.array(y_hat_aggregated)
            y_hat.append(np.argmax(np.average(y_hat_aggregated, axis=0), axis=0))
            y.append(self.filename_to_label[f])

        y_hat = np.array(y_hat)
        y = np.array(y)

        print(classification_report(y, y_hat, target_names=self.labels, digits=3))
