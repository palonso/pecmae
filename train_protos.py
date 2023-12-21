import pickle as pk
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pytorch_lightning as L
import numpy as np
import torch
import torchmetrics
import yaml
from datasets import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim, nn, utils
from sklearn.cluster import KMeans

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
        protos: int = 1,
        batch_size: int = 32,
    ):
        super().__init__()

        self.time_dim = time_dim
        self.feat_dim = feat_dim
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.protos = protos
        self.n_protos = self.protos.shape[0]

        self.protos = nn.Parameter(data=torch.tensor(self.protos), requires_grad=True)
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

        loss_p = torch.mean(torch.min(distance_reduce, dim=1).values) + torch.mean(
            torch.min(distance_reduce, dim=0).values
        )

        y_hat = self.linear(distance_reduce)

        loss_c = self.xent(y_hat, y)

        # TODO: try to solve classification for now
        loss = loss_p + loss_c
        # loss = loss_c

        self.log(f"{split}_loss", loss, prog_bar=True)

        acc = self.accuracy(y_hat, y)

        self.log(f"{split}_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer


def trim_embeddings(examples: list, timestamps: int = 300, mode: str = "middle"):
    features = []
    for feature in examples["feature"]:
        feature = np.array(feature).squeeze()

        if mode == "middle":
            middle = feature.shape[0] // 2
            feature = feature[middle - timestamps // 2 : middle + timestamps // 2, :]
        elif mode == "random":
            start = np.random.randint(0, feature.shape[0] - timestamps)
            feature = feature[start : start + timestamps, :]
        elif mode == "beggining":
            feature = feature[:timestamps, :]
        else:
            raise ValueError(f"mode {mode} not supported")
        features.append(feature)

    examples["feature"] = features
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
    examples["feature"] = [
        (np.array(feature) - mean) / (2 * std) for feature in examples["feature"]
    ]

    data = np.array(examples["feature"])
    print(f"data min: {np.min(data):.3f}")
    print(f"data max: {np.max(data):.3f}")

    return examples


def label_to_idx(examples: list):
    label_set = list(set(examples["label"]))
    label_set.sort()
    for i, label in enumerate(label_set):
        print(f"label {label}: {i}")
    label_map = {label: i for i, label in enumerate(label_set)}
    examples["label"] = [label_map[label] for label in examples["label"]]

    return examples


def create_protos(
    ds: Dataset,
    protos_init: str,
    shape: tuple,
    n_protos_per_label: int,
    labels: set = None,
):
    """Init prototypes to random weights, class centroids (kmeans) or to the sample closses to the class centroid (kmeans-sample)"""

    n_protos = n_protos_per_label * len(labels)
    kmeans_file = Path(f"kmeans_data_{len(labels)}labels_{n_protos}protos.pk")

    if protos_init == "random":
        protos = torch.randn([n_protos, *shape])
    elif protos_init in ("kmeans-centers", "kmeans-samples"):
        if kmeans_file.exists():
            print("kmeans data file found, loading...")
            with open(kmeans_file, "rb") as handle:
                kmeans_data = pk.load(handle)
        else:
            print("kmeans data file not found, computing...")
            # compute k_means
            ds_np = ds.with_format("numpy")

            kmeans_data = dict()
            for label in list(labels):
                print(f"computing kmeans for label {label}")

                indices = ds_np["label"] == label
                samples = ds_np["feature"][indices]

                # select a slice of timestamps samples to speed up kmeans
                samples = samples.reshape(samples.shape[0], -1)
                kmeans = KMeans(n_clusters=n_protos_per_label, n_init="auto")
                samples_dis = kmeans.fit_transform(samples)
                kmeans_data[f"label.{label}.distances"] = samples_dis
                kmeans_data[f"label.{label}.centers"] = kmeans.cluster_centers_
                kmeans_data[f"label.{label}.samples"] = samples[
                    np.argmin(samples_dis, axis=0)
                ]

            pk.dump(kmeans_data, open(kmeans_file, "wb"))

    else:
        raise ValueError(f"protos_init {protos_init} not supported")

    if protos_init == "kmeans-samples":
        print("using kmeans  closest sample as protos")
        key = "samples"
    elif protos_init == "kmeans-centers":
        key = "centers"

    protos = [kmeans_data[f"label.{label}.{key}"] for label in labels]
    protos = np.hstack(protos)
    protos = protos.reshape(n_protos, *shape)

    return protos


def train(
    data_dir: Path = None,
    metadata_file: Path = None,
    protos_init: str = None,
    n_protos_per_label: int = 1,
    batch_size: int = 32,
):
    print("creating dataset")
    ds = Dataset.from_generator(
        dataset_generator,
        gen_kwargs={
            "metadata_file": metadata_file,
            "data_dir": data_dir,
        },
    )

    # compress rows
    print("cleaning dataset")
    ds = ds.map(trim_embeddings, batched=True, num_proc=32, batch_size=32)

    # onehot encode
    print("cleaning labels")
    ds = ds.map(label_to_idx, batched=True, batch_size=len(ds))

    # norm
    # ds = ds.map(norm, batched=True, batch_size=len(ds))

    # dataset to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ds.with_format("torch", device=device)

    # splits
    print("creating train split")
    ds = ds.train_test_split(test_size=0.1, seed=seed)
    ds_train = ds["train"]
    ds_test = ds["test"]
    print("creating validation split")
    ds_tran_val = ds_train.train_test_split(test_size=0.1, seed=seed)
    ds_train = ds_tran_val["train"]
    ds_val = ds_tran_val["test"]

    time_dim = ds_val["feature"][0].shape[0]
    feat_dim = ds_val["feature"][0].shape[1]
    print(f"time_dim: {time_dim}, feat_dim: {feat_dim}")

    labels = set(ds_val["label"].cpu().numpy())
    n_labels = len(labels)
    n_protos = n_labels * n_protos_per_label
    print(f"n_labels: {n_labels}, n_protos: {n_protos}")

    protos = create_protos(
        ds_train,
        protos_init,
        (time_dim, feat_dim),
        n_protos_per_label=n_protos_per_label,
        labels=labels,
    )

    model = ZinemaNet(
        time_dim=time_dim,
        feat_dim=feat_dim,
        n_classes=n_labels,
        protos=protos,
    )

    loader_train = utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = utils.data.DataLoader(ds_val, batch_size=batch_size)
    loader_test = utils.data.DataLoader(ds_test, batch_size=batch_size)

    logger = TensorBoardLogger("tb_logs", name="zinemanet")

    trainer = L.Trainer(
        max_epochs=200,
        devices=[0],
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
        precision="16-mixed",
        logger=logger,
    )
    trainer.fit(
        model,
        loader_train,
        loader_val,
    )
    trainer.test(model, loader_test)
    lin_weights = model.linear.weight.detach().cpu().numpy()
    protos = model.protos.detach().cpu().numpy()

    # Get current timestamp
    current_timestamp = datetime.now()
    formatted_timestamp = current_timestamp.strftime("%y%m%d_%H%M%S")

    np.save(f"lin_weights_{formatted_timestamp}.npy", lin_weights)
    np.save(f"protos_{formatted_timestamp}.npy", protos)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument(
        "--protos-init", choices=["random", "kmeans-centers", "kmeans-samples"]
    )
    parser.add_argument("--n-protos-per-label", type=int, default=1)

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        protos_init=args.protos_init,
        n_protos_per_label=args.n_protos_per_label,
    )
