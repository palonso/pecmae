import csv
import pickle as pk
import random
from argparse import ArgumentParser

# from datetime import datetime
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
from labelmaps import (
    gtzan_label2id,
    nsynth_label2id,
    xaigenre_label2id,
    medley_solos_label2id,
)


def get_labelmap(dataset: str):
    if dataset == "gtzan":
        return gtzan_label2id
    elif dataset == "nsynth":
        return nsynth_label2id
    elif dataset == "xai_genre":
        return xaigenre_label2id
    elif dataset == "medley_solos":
        return medley_solos_label2id
    else:
        raise ValueError(f"dataset {dataset} not supported")


def label2id(examples: dict, labelmap: dict):
    if isinstance(examples["label"], list):
        examples["label"] = [labelmap[label] for label in examples["label"]]
    else:
        examples["label"] = labelmap[examples["label"]]
    return examples


def dataset_generator(
    metadata_file: Path,
    data_dir: Path,
    dataset: str,
    seed: int = None,
):
    if dataset in ("nsynth"):
        with open(metadata_file, "r") as f:
            metadata = yaml.load(f, Loader=yaml.SafeLoader)
        for k, v in metadata["groundTruth"].items():
            feature_file = (data_dir / k).with_suffix(".npy")
            feature = np.load(feature_file)

            yield {
                "feature": feature,
                "label": v,
                "filename": k,
            }

    elif dataset in ("xai_genre", "medley_solos", "gtzan"):
        with open(metadata_file, "r") as f:
            metadata = csv.reader(f, delimiter="\t")
            metadata = list(metadata)

            if seed:
                random.seed(seed)
            random.shuffle(metadata)

            for genre, sid in metadata:
                feature_file = (data_dir / sid).with_suffix(".npy")
                if not feature_file.exists():
                    print(f"file {feature_file} does not exist")
                    continue
                feature = np.load(feature_file)

                yield {
                    "feature": feature,
                    "label": genre,
                    "filename": sid,
                }


lambd = 1.0


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        global lambd
        # return grad_output.neg()
        return grad_output * -lambd


def grad_reverse(x):
    return GradReverse.apply(x)


class domain_classifier_mlp(nn.Module):
    def __init__(self, input_dim=1200):
        super(domain_classifier_mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(100, 1)
        self.drop = nn.Dropout1d(0.25)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return x


class domain_classifier_conv(nn.Module):
    def __init__(self):
        super(domain_classifier_conv, self).__init__()

        self.cv1 = torch.nn.Conv2d(1, 16, 5)
        self.mp1 = torch.nn.MaxPool2d(2)
        self.cv2 = torch.nn.Conv2d(16, 16, 5)
        self.mp2 = torch.nn.MaxPool2d(2)
        self.cv3 = torch.nn.Conv2d(16, 32, 5)
        self.mp3 = torch.nn.MaxPool2d(4)
        self.cv4 = torch.nn.Conv2d(32, 32, 5)
        self.mp4 = torch.nn.MaxPool2d(4)
        self.ln1 = torch.nn.Linear(960, 1)

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
        proto_loss: str = "l2",
        proto_loss_samples: str = "any_sample",
        use_discriminator: bool = False,
        discriminator_type: str = "mlp",
        distance: str = "l2",
        freeze_protos: bool = False,
        time_summarization: str = None,
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
                num_layers=4,
                dropout=0.2,
                batch_first=True,
            )

    def training_step(self, batch, batch_idx, split="train"):
        x = batch["feature"]
        y = batch["label"]

        # flatten time and embedding dimension
        if self.time_summarization is None:
            x = x.flatten(1)
            protos = self.protos.flatten(1)

        elif self.time_summarization == "lstm":
            protos = self.time_summarizer(self.protos)[0].flatten(1)
            x = x.flatten(1)

        optimizers = self.optimizers()
        lr_schedulers = self.lr_schedulers()

        if self.use_discriminator:
            optimizer_c = optimizers[0]
            lr_scheduler_c = lr_schedulers[0]

            optimizer_d = optimizers[1]
            lr_scheduler_d = lr_schedulers[1]
        else:
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

        self.log(f"{split}_distance", distance.mean())

        y_hat = self.linear(similarity)
        acc = self.accuracy(y_hat, y)

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

        if split == "train":
            self.manual_backward(loss)
            optimizer_c.step()

            lr_scheduler_c.step()
            self.log(f"{split}_lr_class", lr_scheduler_c.get_last_lr()[0])

        self.log(f"{split}_acc", acc, prog_bar=True)
        self.log(f"{split}_class_loss", loss_c)
        self.log(f"{split}_proto_loss", loss_p)

        if self.use_discriminator:
            optimizer_d.zero_grad()

            p = float(self.i) / self.total_steps
            global lambd

            lambd = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1

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

            self.log(f"{split}_disc_acc", acc_disc)

            if split == "train":
                self.log("lambda", lambd)
                self.manual_backward(loss_d)
                optimizer_d.step()

                lr_scheduler_d.step()
                self.log(f"{split}_lr_disc", lr_scheduler_d.get_last_lr()[0])

            self.log(f"{split}_disc_loss", loss_d)

        if split == "train":
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

        if self.use_discriminator:
            optimizer_d = optim.Adam(
                self.parameters(),
                weight_decay=self.weight_decay,
            )
            scheduler_d = optim.lr_scheduler.OneCycleLR(
                optimizer_d,
                max_lr=self.max_lr,
                total_steps=self.total_steps,
            )
            optimizers.append(
                {"optimizer": optimizer_d, "lr_scheduler": scheduler_d},
            )

        return optimizers

    def save_checkpoint(self):
        if self.time_summarization:
            with torch.no_grad():
                protos = self.time_summarizer(self.protos)[0]
        else:
            protos = self.protos

        protos = protos.detach().cpu().numpy()

        out_data_dir = Path("out_data/")
        out_data_dir.mkdir(exist_ok=True)

        protos_file = out_data_dir / f"protos_v{self.logger.version}_s{self.i}.npy"
        print(f"saving protos to {protos_file}")

        np.save(protos_file, protos)


def trim_embeddings(examples: list, timestamps: int = 300, mode: str = "middle"):
    features = []
    labels = []
    filenames = []

    if not isinstance(examples["label"], list):
        examples["feature"] = [examples["feature"]]
        examples["label"] = [examples["label"]]
        examples["filename"] = [examples["filename"]]

    for feature, label, filename in zip(
        examples["feature"], examples["label"], examples["filename"]
    ):
        feature = np.array(feature)

        if len(feature.shape) == 3:
            if mode == "middle":
                middle = feature.shape[0] // 2
                feature = feature[middle]
            elif mode == "random":
                index = np.random.randint(0, feature.shape[0])
                feature = feature[index]
            elif mode == "beginning":
                feature = feature[0]
            elif mode == "all":
                labels.extend([label] * feature.shape[0])
                filenames.extend([filename] * feature.shape[0])
            else:
                raise ValueError(f"mode {mode} not supported")

        elif len(feature.shape) == 2:
            if mode == "middle":
                middle = feature.shape[0] // 2
                feature = feature[
                    middle - timestamps // 2 : middle + timestamps // 2, :
                ]
                feature = np.expand_dims(feature, 0)
            elif mode == "random":
                start = np.random.randint(0, feature.shape[0] - timestamps)
                feature = feature[start : start + timestamps, :]
                feature = np.expand_dims(feature, 0)
            elif mode == "beginning":
                feature = feature[:timestamps, :]
                feature = np.expand_dims(feature, 0)
            elif mode == "all":
                n_chunks = max(feature.shape[0] // timestamps, 1)
                feature = feature[: n_chunks * timestamps, :].reshape(
                    -1, timestamps, feature.shape[-1]
                )
                labels.extend([label] * n_chunks)
                filenames.extend([filename] * n_chunks)
            else:
                raise ValueError(f"mode {mode} not supported")

        features.append(feature)

    if mode == "all":
        examples["label"] = labels
        examples["filename"] = filenames

    features = np.vstack(features)
    if features.shape[0] == 1:
        features = features.squeeze(0)
    features = features.tolist()

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
    data_dir: Path = None,
):
    """Init prototypes to random weights, class centroids (kmeans) or to the sample closses to the class centroid (kmeans-sample)"""

    n_protos = n_protos_per_label * len(labels)

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

    elif protos_init == "proto-dict":
        with open(proto_file, "r") as f:
            proto_dict = yaml.safe_load(f)

        protos = []
        for label in labels:
            candidate_protos = proto_dict[label]
            for i in range(n_protos_per_label):
                proto_data_file = (data_dir / candidate_protos[i]).with_suffix(".npy")
                proto = np.load(proto_data_file)

                # TODO how to use the whole proto?
                # taking the chunk in the middle for now
                proto = proto[proto.shape[0] // 2]

                protos.append(proto)

        protos = np.array(protos)
        print("protos shape", protos.shape)

    elif protos_init in ("kmeans-centers", "kmeans-samples"):
        # compute k_means
        ds_np = ds.with_format("numpy")

        kmeans_data = dict()
        for label in list(labels):
            print(f"computing kmeans for label {label}")

            indices = ds_np["label"] == label
            samples = ds_np["feature"][indices.squeeze()]

            # select a slice of timestamps samples to speed up kmeans
            samples = samples.reshape(samples.shape[0], -1)
            kmeans = KMeans(n_clusters=n_protos_per_label, n_init="auto")
            samples_dis = kmeans.fit_transform(samples)
            kmeans_data[f"label.{label}.distances"] = samples_dis
            kmeans_data[f"label.{label}.centers"] = kmeans.cluster_centers_
            kmeans_data[f"label.{label}.samples"] = samples[
                np.argmin(samples_dis, axis=0)
            ]

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
    proto_loss: str = "l2",
    proto_loss_samples: str = "class",
    use_discriminator: bool = False,
    discriminator_type: str = "mlp",
    checkpoint: Path = None,
    dataset: str = None,
    freeze_protos: bool = False,
    time_summarization: str = None,
):
    hyperparams = locals()

    print("creating dataset")
    ds = Dataset.from_generator(
        dataset_generator,
        writer_batch_size=500,
        gen_kwargs={
            "metadata_file": metadata_file,
            "data_dir": data_dir,
            "dataset": dataset,
        },
    )

    # trim embeddings, select beginning, middle or random chunk
    print("trimming embeddings")
    ds = ds.map(
        trim_embeddings,
        num_proc=32,
        batched=True,
        batch_size=32,
        fn_kwargs={"timestamps": timestamps, "mode": trim_mode},
    )

    print("label encoding")
    ds = ds.map(
        label2id,
        fn_kwargs={"labelmap": get_labelmap(dataset)},
    )

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

    labels = set(get_labelmap(dataset).keys())
    n_labels = len(labels)
    n_protos = n_labels * n_protos_per_label
    print(f"n_labels: {n_labels}, n_protos: {n_protos}")

    protos = create_protos(
        ds_train,
        protos_init,
        (time_dim, feat_dim),
        n_protos_per_label=n_protos_per_label,
        labels=list(range(len(labels))),
        proto_file=proto_file,
        data_dir=data_dir,
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

    if checkpoint is not None:
        print(f"loading checkpoint from {checkpoint}")
        model = ZinemaNet.load_from_checkpoint(
            checkpoint,
            protos=protos,
            time_dim=time_dim,
            feat_dim=feat_dim,
            n_labels=n_labels,
            batch_size=batch_size,
            total_steps=len(loader_train) * epochs,
            temp=temp,
            alpha=alpha,
            max_lr=max_lr,
            proto_loss=proto_loss,
            proto_loss_samples=proto_loss_samples,
            use_discriminator=use_discriminator,
            discriminator_type=discriminator_type,
            freeze_protos=freeze_protos,
        )
    else:
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
            proto_loss=proto_loss,
            proto_loss_samples=proto_loss_samples,
            use_discriminator=use_discriminator,
            discriminator_type=discriminator_type,
            freeze_protos=freeze_protos,
            time_summarization=time_summarization,
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
    if model.time_summarization:
        with torch.no_grad():
            protos = model.time_summarizer(model.protos)[0].flatten(1, 2)
    else:
        protos = model.protos
    protos = protos.detach().cpu().numpy()

    # current_timestamp = datetime.now()
    # formatted_timestamp = current_timestamp.strftime("%y%m%d_%H%M%S")

    out_data_dir = Path("out_data")
    out_data_dir.mkdir(exist_ok=True)

    np.save(out_data_dir / f"lin_weights_v{logger.version}.npy", lin_weights)
    np.save(out_data_dir / f"protos_v{logger.version}.npy", protos)


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
            "proto-dict",
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
    parser.add_argument("--proto-loss", default="l2", choices=["l1", "l2", "info_nce"])
    parser.add_argument("--proto-loss-samples", default="all", choices=["all", "class"])
    parser.add_argument("--use-discriminator", action="store_true")
    parser.add_argument("--discriminator-type", default="mlp", choices=["mlp", "conv"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--dataset", type=str, choices=["gtzan", "nsynth", "xai_genre", "medley_solos"]
    )
    parser.add_argument("--freeze-protos", action="store_true")
    parser.add_argument(
        "--time-summarization",
        type=str,
        default=None,
        choices=[None, "lstm", "attention"],
    )

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
        gpu_id=args.gpu_id,
        temp=args.temp,
        alpha=args.alpha,
        proto_loss_samples=args.proto_loss_samples,
        proto_loss=args.proto_loss,
        use_discriminator=args.use_discriminator,
        discriminator_type=args.discriminator_type,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        time_summarization=args.time_summarization,
    )
