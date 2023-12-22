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

from similarities import BilinearSimilarity, InfoNCE

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
        protos: np.array,
        time_dim: int = 300,
        feat_dim: int = 768,
        n_labels: int = 10,
        batch_size: int = 32,
        total_steps: int = 1000,
        weight_decay: float = 1e-4,
        max_lr: float = 1e-3,
        temp: float = 0.1,
        alpha: float = 0.5,
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

        self.n_protos = self.protos_weights.shape[0]
        self.n_protos_per_label = self.n_protos // self.n_labels

        self.protos = nn.Parameter(
            data=torch.tensor(self.protos_weights), requires_grad=True
        )
        self.linear = torch.nn.Linear(self.n_protos, self.n_labels)

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

        # to use the scheduler
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx, split="train"):
        x = batch["feature"]
        y = batch["label"]

        # flatten time and embedding dimension
        x = x.flatten(1, 2)
        protos = self.protos.flatten(1, 2)

        if split == "train":
            optimizer = self.optimizers()
            optimizer.zero_grad()

        similarity = self.info_nce(x, protos, output="logits")

        distance = torch.exp(-similarity)
        self.log(f"{split}_distance", distance.mean())

        y_hat = self.linear(similarity)
        acc = self.accuracy(y_hat, y)

        # classification loss
        loss_c = self.xent(y_hat / self.temp, y)
        # prototype loss
        loss_p = torch.mean(torch.min(distance, dim=0).values)

        loss = self.alpha * loss_p + (1 - self.alpha) * loss_c

        if split == "train":
            self.manual_backward(loss)
            optimizer.step()

            scheduler = self.lr_schedulers()
            scheduler.step()
            self.log(f"{split}_lr", scheduler.get_last_lr()[0])

        self.log(f"{split}_loss", loss, prog_bar=True)
        self.log(f"{split}_acc", acc, prog_bar=True)
        self.log(f"{split}_proto_loss", loss_p)
        self.log(f"{split}_class_loss", loss_c)

        return loss

    def validation_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=self.total_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


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
    proto_file: Path = None,
):
    """Init prototypes to random weights, class centroids (kmeans) or to the sample closses to the class centroid (kmeans-sample)"""

    n_protos = n_protos_per_label * len(labels)
    kmeans_file = Path(f"kmeans_data_{len(labels)}labels_{n_protos}protos.pk")

    if protos_init == "random":
        protos = torch.randn([n_protos, *shape])

    elif protos_init == "proto-file":
        protos = np.load(proto_file)
        print(f"protos loaded from {proto_file}")
        protos = torch.tensor(protos)
        assert protos.shape == (
            n_protos,
            *shape,
        ), f"protos shape mismatch. found: {protos.shape} expected: {(n_protos, *shape)}"

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

        if protos_init == "kmeans-samples":
            print("using kmeans  closest sample as protos")
            key = "samples"
        elif protos_init == "kmeans-centers":
            key = "centers"

        protos = [kmeans_data[f"label.{label}.{key}"] for label in labels]
        protos = np.hstack(protos)
        protos = protos.reshape(n_protos, *shape)

    else:
        raise ValueError(f"protos_init {protos_init} not supported")

    return protos


def train(
    data_dir: Path = None,
    metadata_file: Path = None,
    protos_init: str = None,
    n_protos_per_label: int = 1,
    batch_size: int = 32,
    seed: int = 42,
    epochs: int = 1000,
    max_lr: float = 1e-3,
    val_test_ratio: float = 0.1,
    timestamps: int = 300,
    trim_mode: str = "middle",
    gpu_id: int = 0,
    proto_file: Path = None,
    temp: float = 0.1,
    alpha: float = 0.5,
):
    hyperparams = locals()

    print("creating dataset")
    ds = Dataset.from_generator(
        dataset_generator,
        gen_kwargs={
            "metadata_file": metadata_file,
            "data_dir": data_dir,
        },
    )

    # trim embeddings, select beggining, middle or random chunk
    print("trimming embeddings")
    ds = ds.map(
        trim_embeddings,
        batched=True,
        num_proc=32,
        batch_size=32,
        fn_kwargs={"timestamps": timestamps, "mode": trim_mode},
    )

    print("label encoding")
    ds = ds.map(label_to_idx, batched=True, batch_size=len(ds))

    # TODO: check if norm is needed
    # ds = ds.map(norm, batched=True, batch_size=len(ds))

    # dataset to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ds.with_format("torch", device=device)

    print("creating train splits")
    ds = ds.train_test_split(test_size=val_test_ratio, seed=seed)
    ds_train = ds["train"]
    ds_test = ds["test"]
    ds_tran_val = ds_train.train_test_split(test_size=val_test_ratio, seed=seed)
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
        proto_file=proto_file,
    )

    loader_train = utils.data.DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=20,
    )
    loader_val = utils.data.DataLoader(
        ds_val,
        batch_size=batch_size,
        # num_workers=20,
    )
    loader_test = utils.data.DataLoader(
        ds_test,
        batch_size=batch_size,
        # num_workers=10,
    )

    model = ZinemaNet(
        protos,
        time_dim=time_dim,
        feat_dim=feat_dim,
        n_labels=n_labels,
        batch_size=batch_size,
        total_steps=len(loader_train) * epochs,
        temp=temp,
        alpha=alpha,
        max_lr=max_lr,
    )

    logger = TensorBoardLogger(
        "tb_logs",
        name="zinemanet",
    )

    logger.log_hyperparams(hyperparams)

    trainer = L.Trainer(
        max_epochs=epochs,
        devices=[gpu_id],
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

    # save parameters
    lin_weights = model.linear.weight.detach().cpu().numpy()
    protos = model.protos.detach().cpu().numpy()

    current_timestamp = datetime.now()
    formatted_timestamp = current_timestamp.strftime("%y%m%d_%H%M%S")

    np.save(f"lin_weights_{formatted_timestamp}.npy", lin_weights)
    np.save(f"protos_{formatted_timestamp}.npy", protos)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument(
        "--protos-init",
        choices=[
            "random",
            "kmeans-centers",
            "kmeans-samples",
            "proto-file",
        ],
    )
    parser.add_argument("--n-protos-per-label", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--proto-file", type=Path, default=None)
    parser.add_argument("--val-test-ratio", type=float, default=0.1)
    parser.add_argument("--timestamps", type=int, default=300)
    parser.add_argument("--trim-mode", type=str, default="middle")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.7)

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        protos_init=args.protos_init,
        n_protos_per_label=args.n_protos_per_label,
        batch_size=args.batch_size,
        seed=args.seed,
        epochs=args.epochs,
        max_lr=args.max_lr,
        proto_file=args.proto_file,
        val_test_ratio=args.val_test_ratio,
        timestamps=args.timestamps,
        trim_mode=args.trim_mode,
        temp=args.temp,
        alpha=args.alpha,
    )
