from argparse import ArgumentParser

from pathlib import Path

import pytorch_lightning as L
import numpy as np
import torch
from datasets import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import utils

from train_protos import (
    dataset_generator,
    ZinemaNet,
    create_protos,
    trim_embeddings,
    get_labelmap,
    label2id,
)
from labelmaps import gtzan_label2id, nsynth_label2id, xaigenre_label2id


def norm(sample, mean, std):
    sample["feature"] = (sample["feature"] - mean) / (std * 2)
    return sample


def get_dataset(
    metadata_file: Path,
    data_dir: Path,
    dataset: str,
    timestamps: int,
    trim_mode: str,
    seed: int = None,
    do_normalization: bool = False,
    ds_mean: float = None,
    ds_std: float = None,
):
    ds = Dataset.from_generator(
        dataset_generator,
        gen_kwargs={
            "metadata_file": metadata_file,
            "data_dir": data_dir,
            "dataset": dataset,
            "seed": seed,
        },
    )
    ds = ds.map(
        trim_embeddings,
        fn_kwargs={"timestamps": timestamps, "mode": trim_mode},
        batched=True,
    )
    ds = ds.map(
        label2id,
        fn_kwargs={"labelmap": get_labelmap(dataset)},
    )
    if do_normalization:
        if not ds_std or not ds_mean:
            print("computing dataset stats...")
            features = ds.with_format("numpy")["feature"].squeeze()
            ds_mean = np.mean(features.mean(axis=1))
            ds_std = np.mean(features.std(axis=1))
            print(f"mean: {ds_mean} std: {ds_std}")

        ds = ds.map(norm, fn_kwargs={"mean": ds_mean, "std": ds_std})

    ds = ds.with_format("torch")
    return ds, ds_mean, ds_std


def train(
    data_dir: Path = None,
    data_dir_test: Path = None,
    metadata_file_train: Path = None,
    metadata_file_val: Path = None,
    metadata_file_test: Path = None,
    protos_init: str = None,
    n_protos_per_label: int = 1,
    batch_size: int = 32,
    seed: int = 42,
    total_steps=30000,
    max_lr: float = 1e-3,
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
    time_summarization: str = "none",
    do_normalization: bool = False,
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

    # time_dim = ds_val[0]["feature"].shape[0]
    time_dim = timestamps
    # feat_dim = ds_val[0]["feature"].shape[1]
    feat_dim = 768
    print(f"time_dim: {time_dim}, feat_dim: {feat_dim}")

    labels = list(get_labelmap(dataset).keys())
    labels.sort()
    n_labels = len(labels)
    n_protos = n_labels * n_protos_per_label
    print(f"n_labels: {n_labels}, n_protos: {n_protos}")

    protos = create_protos(
        ds_train,
        protos_init,
        (time_dim, feat_dim),
        n_protos_per_label=n_protos_per_label,
        labels=list(range(n_labels)),
        proto_file=proto_file,
        data_dir=Path("feats_xai_genre_v2_protos/base/audio/"),
    )

    loader_train = utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = utils.data.DataLoader(ds_val, batch_size=batch_size)
    loader_test = utils.data.DataLoader(ds_test, batch_size=batch_size)

    if checkpoint is not None:
        print(f"loading checkpoint from {checkpoint}")
        model = ZinemaNet.load_from_checkpoint(
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
            time_summarization=time_summarization,
            do_normalization=do_normalization,
            ds_mean=ds_mean,
            ds_std=ds_std,
            labels=labels,
        )
    else:
        model = ZinemaNet(
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
            do_normalization=do_normalization,
            ds_mean=ds_mean,
            ds_std=ds_std,
            labels=labels,
        )

    logger = TensorBoardLogger(
        "tb_logs",
        name="zinemanet",
    )

    logger.log_hyperparams(hyperparams)

    checkpoint_callback = ModelCheckpoint(monitor="val_acc_aggregated", mode="max")

    trainer = L.Trainer(
        max_steps=total_steps,
        devices=[gpu_id],
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model,
        loader_train,
        loader_val,
    )

    trainer.test(model, loader_test, ckpt_path=checkpoint_callback.best_model_path)

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

    if do_normalization:
        protos = (protos * ds_std * 2) + ds_mean

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
    parser.add_argument("--total-steps", type=int, default=30000)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--proto-file", type=Path, default=None)
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
    parser.add_argument(
        "--time-summarization",
        type=str,
        default="none",
        choices=["none", "lstm", "transformer", "dense_res"],
    )
    parser.add_argument("--do-normalization", type=lambda x: x == "True", default=False)

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
        total_steps=args.total_steps,
        max_lr=args.max_lr,
        proto_file=args.proto_file,
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
        freeze_protos=args.freeze_protos,
        time_summarization=args.time_summarization,
        do_normalization=args.do_normalization,
    )
