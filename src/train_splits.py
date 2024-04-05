from argparse import ArgumentParser

from pathlib import Path

import pytorch_lightning as L
import numpy as np
from datasets import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import utils

from train_protos import (
    dataset_generator,
    trim_embeddings,
    get_labelmap,
    label2id,
)
from models import MLP


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
    batch_size: int = 32,
    seed: int = 42,
    total_steps=30000,
    max_lr: float = 1e-3,
    trim_mode: str = "middle",
    gpu_id: int = 0,
    checkpoint: Path = None,
    dataset: str = None,
    do_normalization: bool = False,
):
    hyperparams = locals()

    if data_dir_test is None:
        data_dir_test = data_dir

    ds_train, ds_mean, ds_std = get_dataset(
        metadata_file_train,
        data_dir,
        dataset,
        1,
        trim_mode,
        seed=seed,
        do_normalization=do_normalization,
    )

    if metadata_file_val:
        ds_val, _, _ = get_dataset(
            metadata_file_val,
            data_dir,
            dataset,
            1,
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
            1,
            trim_mode,
            do_normalization=do_normalization,
            ds_mean=ds_mean,
            ds_std=ds_std,
        )
    else:
        print(
            "warning: test metadata file not detected. using 0.15 of train as test split"
        )
        ds_split = ds_train.train_test_split(test_size=0.15)
        ds_train = ds_split["train"]
        ds_test = ds_split["test"]

    time_dim = 1
    feat_dim = ds_val[0]["feature"].shape[-1]
    print(f"time_dim: {time_dim}, feat_dim: {feat_dim}")

    labels = list(get_labelmap(dataset).keys())
    labels.sort()
    n_labels = len(labels)
    print(f"n_labels: {n_labels}")

    loader_train = utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = utils.data.DataLoader(ds_val, batch_size=batch_size)
    loader_test = utils.data.DataLoader(ds_test, batch_size=batch_size)

    if checkpoint is not None:
        print(f"loading checkpoint from {checkpoint}")
        model = MLP.load_from_checkpoint(
            checkpoint,
            in_size=time_dim * feat_dim,
            out_size=n_labels,
            labels=labels,
        )
    else:
        model = MLP(
            in_size=time_dim * feat_dim,
            out_size=n_labels,
            labels=labels,
            lr=max_lr,
        )

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    trainer = L.Trainer(
        max_steps=total_steps,
        devices=[gpu_id],
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
        precision="16-mixed",
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model,
        loader_train,
        loader_val,
    )

    trainer.test(model, loader_test, ckpt_path=checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-dir-test", type=Path, required=False)
    parser.add_argument("--metadata-file-train", type=Path, required=True)
    parser.add_argument("--metadata-file-val", type=Path)
    parser.add_argument("--metadata-file-test", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=30000)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--trim-mode", type=str, default="middle")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "gtzan",
            "nsynth",
            "xai_genre",
            "medley_solos",
        ],
    )
    parser.add_argument("--do-normalization", type=lambda x: x == "True", default=False)

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        data_dir_test=args.data_dir_test,
        metadata_file_train=args.metadata_file_train,
        metadata_file_val=args.metadata_file_val,
        metadata_file_test=args.metadata_file_test,
        batch_size=args.batch_size,
        seed=args.seed,
        total_steps=args.total_steps,
        max_lr=args.max_lr,
        trim_mode=args.trim_mode,
        gpu_id=args.gpu_id,
        dataset=args.dataset,
        do_normalization=args.do_normalization,
    )
