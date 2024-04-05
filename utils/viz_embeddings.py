from argparse import ArgumentParser
from pathlib import Path

import numpy as np

matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import umap
from datasets import Dataset


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


def average_codes(examples: list):
    examples["feature"] = [
        np.array(feature).squeeze().mean(axis=0)[:2245]
        for feature in examples["feature"]
    ]
    return examples


def average(examples: list):
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


def viz_embeddings(data_dir: Path = None, metadata_file: Path = None):
    ds = Dataset.from_generator(
        dataset_generator,
        gen_kwargs={
            "metadata_file": metadata_file,
            "data_dir": data_dir,
        },
    )

    # compress rows

    ds = ds.map(average, batched=True, batch_size=64)

    # norm
    ds = ds.map(norm, batched=True, batch_size=len(ds))

    ds = ds.with_format("numpy")

    x = np.array(ds["feature"])
    y = np.array(ds["label"])

    print("Projecting to 2D with UMAP")
    projector = umap.UMAP()
    projected = projector.fit_transform(x)

    x_x = projected[:, 0]
    x_y = projected[:, 1]

    plt.title("2D projection of features")
    sns.scatterplot(x=x_x, y=x_y, hue=y)

    plt.savefig("umap.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)

    args = parser.parse_args()
    viz_embeddings(data_dir=args.data_dir, metadata_file=args.metadata_file)
