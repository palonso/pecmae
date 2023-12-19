from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as L
import numpy as np
import torch
import torchmetrics
import yaml
from datasets import Dataset
from torch import optim, nn, utils

seed = 42


def dataset_generator(metadata_file: Path, data_dir: Path):
    with open(metadata_file, "r") as f:
        metadata = yaml.load(f, Loader=yaml.SafeLoader)
    for k, v in metadata["groundTruth"].items():
        feature_file = (data_dir / k).with_suffix(".npy")
        feature = np.load(feature_file)

        yield {
            "feature": feature,
            "label": v,
        }


class ZinemaNet(L.LightningModule):
    def __init__(
        self,
        time_dim: int = 300,
        feat_dim: int = 768,
        n_classes: int = 10,
        hidden_size: int = 128,
        protos_per_class: int = 1,
        batch_size: int = 32,
    ):
        super().__init__()

        self.time_dim = time_dim
        self.feat_dim = feat_dim
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.protos_per_class = protos_per_class
        self.n_protos = self.n_classes * self.protos_per_class

        protos_weights = torch.randn([self.n_protos, self.time_dim, self.feat_dim])
        self.protos = nn.Parameter(data=protos_weights, requires_grad=True)
        print("protos_shape", self.protos.shape)
        self.linear = torch.nn.Linear(self.n_protos, self.n_classes)

        # self.l2 = nn.MSELoss()
        self.xent = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=n_classes,
        )

    def training_step(self, batch, batch_idx, split="train"):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["feature"]
        y = batch["label"]

        batch_size = x.shape[0]
        n_protos = self.protos.shape[0]

        x = x.unsqueeze(1).repeat(1, self.n_protos, 1, 1)
        # print("x_shape", x.shape)
        protos_batch = (
            self.protos.unsqueeze(0).repeat(x.shape[0], 1, 1, 1).to(self.device)
        )
        # print("protos_batch_shape", protos_batch.shape)

        distance = torch.cdist(x, protos_batch)
        # print("distance_shape", distance.shape)

        # todo: improve
        distance_reduce = distance.mean(dim=[2, 3])
        # print("distance_reduce_shape", distance_reduce.shape)

        loss_p = (
            torch.sum(torch.min(distance_reduce, dim=1).values) / batch_size
            + torch.sum(torch.min(distance_reduce, dim=0).values) / n_protos
        )

        y_hat = self.linear(distance_reduce)

        loss_c = self.xent(y_hat, y)

        loss = loss_p + loss_c

        self.log(
            f"{split}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer


def trim_embeddings(examples: list):
    examples["feature"] = [
        np.array(feature).squeeze()[:2245, :] for feature in examples["feature"]
    ]
    return examples


def average_encodecmae(examples: list):
    examples["feature"] = [
        np.array(feature).squeeze().mean(axis=0) for feature in examples["feature"]
    ]
    return examples


def norm(examples: list):
    data = np.array(examples["feature"])
    mean = np.mean(data)
    std = np.std(data)
    print("mean", mean)
    print("std", std)
    examples["feature"] = [
        (np.array(feature) - mean) / (2 * std) for feature in examples["feature"]
    ]

    data = np.array(examples["feature"])
    print(f"data min: {np.min(data):.3f}")
    print(f"data max: {np.max(data):.3f}")

    return examples


def label_to_idx(examples: list):
    label_set = list(set(examples["label"]))
    label_map = {label: i for i, label in enumerate(label_set)}
    examples["label"] = [label_map[label] for label in examples["label"]]

    return examples


def train(data_dir: Path = None, metadata_file: Path = None):
    ds = Dataset.from_generator(
        dataset_generator,
        gen_kwargs={
            "metadata_file": metadata_file,
            "data_dir": data_dir,
        },
    )

    # compress rows
    ds = ds.map(trim_embeddings, batched=True, num_proc=32, batch_size=32)

    # onehot encode
    ds = ds.map(label_to_idx, batched=True, batch_size=len(ds))

    # norm
    # ds = ds.map(norm, batched=True, batch_size=len(ds))

    # put dataset to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ds.with_format("torch", device=device)

    ds = ds.train_test_split(test_size=0.1, seed=seed)
    ds_train = ds["train"]
    ds_test = ds["test"]
    # ds_tran_val = ds_train.train_test_split(test_size=0.1, seed=seed)
    # ds_train = ds_tran_val["train"]
    # ds_val = ds_tran_val["test"]
    ds_val = ds_test

    time_dim = ds_train["feature"][0].shape[0]
    feat_dim = ds_train["feature"][0].shape[1]

    mlp = ZinemaNet(
        time_dim=time_dim,
        feat_dim=feat_dim,
        n_classes=10,
        protos_per_class=1,
    )

    loader_train = utils.data.DataLoader(ds_train, batch_size=8, shuffle=True)
    loader_val = utils.data.DataLoader(ds_val, batch_size=16)
    loader_test = utils.data.DataLoader(ds_test, batch_size=16)

    trainer = L.Trainer(
        max_epochs=100,
        devices=[0],
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
        precision="16-mixed",
    )
    trainer.fit(
        mlp,
        loader_train,
        loader_val,
    )
    trainer.test(mlp, loader_test)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
    )
