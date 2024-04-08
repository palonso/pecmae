from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pytorch_lightning as L
import numpy as np
import torch
import yaml
from datasets import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from torch import utils
from sklearn.cluster import KMeans

from labelmaps import get_labelmap
from models import PrototypeNet
from commons import get_dataset

lambd = 1.0


def create_protos(
    ds: Dataset,
    protos_init: str,
    shape: tuple,
    n_protos_per_label: int,
    labels: Any = None,
    proto_file: Any = None,
    data_dir: Any = None,
):
    """Init prototypes to random weights, class centroids (kmeans) or to the sample closses to the class centroid (kmeans-sample)"""

    n_protos = n_protos_per_label * len(labels)

    if protos_init == "random":
        protos = torch.randn([n_protos, *shape])

    elif protos_init == "proto-file":
        protos = np.load(proto_file)
        print(f"protos loaded from {proto_file}")
        protos = torch.tensor(protos)
        assert (
            protos.shape == (n_protos, *shape)
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
    data_dir: Any = None,
    data_dir_test: Any = None,
    metadata_file_train: Any = None,
    metadata_file_val: Any = None,
    metadata_file_test: Any = None,
    protos_init: Any = None,
    n_protos_per_label: int = 1,
    batch_size: int = 32,
    seed: int = 42,
    epochs: int = 1000,
    max_lr: float = 1e-3,
    val_test_ratio: float = 0.1,
    timestamps: int = 300,
    trim_mode: str = "middle",
    gpu_id: int = 0,
    proto_file: Any = None,
    temp: float = 0.1,
    alpha: float = 0.5,
    proto_loss: str = "l2",
    proto_loss_samples: str = "class",
    use_discriminator: bool = False,
    discriminator_type: str = "mlp",
    checkpoint: Any = None,
    dataset: Any = None,
    freeze_protos: bool = False,
    time_summarization: Any = None,
    do_normalization: bool = False,
    total_steps=30000,
):
    hyperparams = locals()

    if data_dir_test is None:
        data_dir_test = data_dir

    ds_train, ds_mean, ds_std = get_dataset(
        metadata_file_train,
        data_dir,
        dataset,
        timestamps,
        trim_mode,
        seed=seed,
        do_normalization=do_normalization,
    )

    if metadata_file_val:
        ds_val, _, _ = get_dataset(
            metadata_file_val,
            data_dir,
            dataset,
            timestamps,
            trim_mode,
            seed=seed,
            do_normalization=do_normalization,
            ds_mean=ds_mean,
            ds_std=ds_std,
        )
    else:
        print(
            "warning: validation metadata file not detected. using 0.15 of train as val split"
        )
        ds_split = ds_train.train_test_split(test_size=0.15)
        ds_train = ds_split["train"]
        ds_val = ds_split["test"]

    if metadata_file_test:
        ds_test, _, _ = get_dataset(
            metadata_file_test,
            data_dir_test,
            dataset,
            timestamps,
            trim_mode,
            do_normalization=do_normalization,
            ds_mean=ds_mean,
            ds_std=ds_std,
        )
    else:
        print(
            "warning: test metadata file not detected. using 0.15 of train as val split"
        )
        ds_split = ds_train.train_test_split(test_size=0.15)
        ds_train = ds_split["train"]
        ds_test = ds_split["test"]

    time_dim = timestamps
    # feat_dim = ds_val["feature"][0].shape[1]
    feat_dim = 768
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
    )
    loader_val = utils.data.DataLoader(
        ds_val,
        batch_size=batch_size,
    )
    loader_test = utils.data.DataLoader(
        ds_test,
        batch_size=batch_size,
    )

    if checkpoint is not None:
        print(f"loading checkpoint from {checkpoint}")
        model = PrototypeNet.load_from_checkpoint(
            checkpoint,
            protos=protos,
            time_dim=time_dim,
            feat_dim=feat_dim,
            n_labels=n_labels,
            batch_size=batch_size,
            total_steps=total_steps,
            temp=temp,
            alpha=alpha,
            max_lr=max_lr,
            proto_loss=proto_loss,
            proto_loss_samples=proto_loss_samples,
            use_discriminator=use_discriminator,
            discriminator_type=discriminator_type,
            freeze_protos=freeze_protos,
            labels=labels,
        )
    else:
        model = PrototypeNet(
            protos,
            time_dim=time_dim,
            feat_dim=feat_dim,
            n_labels=n_labels,
            batch_size=batch_size,
            total_steps=total_steps,
            temp=temp,
            alpha=alpha,
            max_lr=max_lr,
            proto_loss=proto_loss,
            proto_loss_samples=proto_loss_samples,
            use_discriminator=use_discriminator,
            discriminator_type=discriminator_type,
            freeze_protos=freeze_protos,
            time_summarization=time_summarization,
            labels=labels,
        )

    logger = TensorBoardLogger(
        "tb_logs",
        name="zinemanet",
    )

    logger.log_hyperparams(hyperparams)

    trainer = L.Trainer(
        max_steps=total_steps,
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
    if model.time_summarization != "none":
        with torch.no_grad():
            protos = model.time_summarizer(model.protos)

        if model.time_summarization == "lstm":
            protos = protos[0]
    else:
        protos = model.protos
    protos = protos.detach().cpu().numpy()

    out_data_dir = Path("out_data")
    out_data_dir.mkdir(exist_ok=True)

    np.save(out_data_dir / f"lin_weights_v{logger.version}.npy", lin_weights)
    np.save(out_data_dir / f"protos_v{logger.version}.npy", protos)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-dir-test", type=Path, required=False)
    parser.add_argument("--metadata-file-train", type=Path, required=True)
    parser.add_argument("--metadata-file-val", type=Path)
    parser.add_argument("--metadata-file-test", type=Path, required=True)
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
    parser.add_argument(
        "--use-discriminator", type=lambda x: x == "True", default=False
    )
    parser.add_argument(
        "--discriminator-type", default="mlp", choices=["mlp", "conv", "linear"]
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--dataset", type=str, choices=["gtzan", "nsynth", "xai_genre", "medley_solos"]
    )
    parser.add_argument("--freeze-protos", action="store_true")
    parser.add_argument("--do-normalization", type=lambda x: x == "True", default=False)
    parser.add_argument(
        "--time-summarization",
        type=str,
        default=None,
        choices=[None, "lstm", "transformer", "dense_res"],
    )
    parser.add_argument("--total-steps", type=int, default=30000)

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        data_dir_test=args.data_dir_test,
        metadata_file_train=args.metadata_file_train,
        metadata_file_val=args.metadata_file_val,
        metadata_file_test=args.metadata_file_test,
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
        do_normalization=args.do_normalization,
        total_steps=args.total_steps,
    )
