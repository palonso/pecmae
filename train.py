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


class MLP(L.LightningModule):
    def __init__(
        self,
        in_size: int = 2096,
        out_size: int = 10,
        hidden_size: int = 128,
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

    def training_step(self, batch, batch_idx, split="train"):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["feature"]
        y = batch["label"]

        y_hat = self.MLP(x)
        loss = self.loss(y_hat, y)
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


def average_encodec(examples: list):
    examples["feature"] = [
        np.array(feature).squeeze().mean(axis=0)[:2245]
        for feature in examples["feature"]
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


def train(
    data_dir: Path = None, metadata_file: Path = None, feature_type: str = "encodec"
):
    ds = Dataset.from_generator(
        dataset_generator,
        gen_kwargs={
            "metadata_file": metadata_file,
            "data_dir": data_dir,
        },
    )

    # compress rows
    if feature_type == "encodec":
        ds = ds.map(average_encodec, batched=True, num_proc=32, batch_size=32)
    elif feature_type == "encodecmae":
        ds = ds.map(average_encodecmae, batched=True, num_proc=32, batch_size=32)

    # onehot encode
    ds = ds.map(label_to_idx, batched=True, batch_size=len(ds))

    # norm
    ds = ds.map(norm, batched=True, batch_size=len(ds))

    # put dataset to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ds.with_format("torch", device=device)

    ds = ds.train_test_split(test_size=0.1, seed=seed)
    ds_train = ds["train"]
    ds_test = ds["test"]
    ds_tran_val = ds_train.train_test_split(test_size=0.1, seed=seed)
    ds_train = ds_tran_val["train"]
    ds_val = ds_tran_val["test"]

    in_size = ds_train["feature"][0].shape[0]

    mlp = MLP(in_size=in_size, out_size=10, hidden_size=128)

    loader_train = utils.data.DataLoader(ds_train, batch_size=32, shuffle=True)
    loader_val = utils.data.DataLoader(ds_val, batch_size=64)
    loader_test = utils.data.DataLoader(ds_test, batch_size=64)

    trainer = L.Trainer(
        max_epochs=10,
        devices=[0],
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(mlp, loader_train, loader_val)
    trainer.test(mlp, loader_test)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument(
        "--feature-type", type=str, required=True, choices=["encodec", "encodecmae"]
    )

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        feature_type=args.feature_type,
    )
