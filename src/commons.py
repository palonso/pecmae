from pathlib import Path
from typing import Any

import csv
import numpy as np
import random
import yaml
from datasets import Dataset

from labelmaps import get_labelmap


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
    seed: Any = None,
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


def trim_embeddings(
    examples: list,
    timestamps: int = 300,
    mode: str = "middle",
):
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


def norm(sample, mean, std):
    sample["feature"] = (sample["feature"] - mean) / (std * 2)
    return sample


def label_to_idx(examples: list):
    label_set = list(set(examples["label"]))
    label_set.sort()
    for i, label in enumerate(label_set):
        print(f"label {label}: {i}")
    label_map = {label: i for i, label in enumerate(label_set)}
    examples["label"] = [label_map[label] for label in examples["label"]]

    return examples


def get_dataset(
    metadata_file: Path,
    data_dir: Path,
    dataset: str,
    timestamps: int,
    trim_mode: str,
    seed: Any = None,
    do_normalization: Any = False,
    ds_mean: Any = None,
    ds_std: Any = None,
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
