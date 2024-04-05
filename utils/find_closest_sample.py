import yaml
from argparse import ArgumentParser
from datasets import Dataset
from pathlib import Path

import numpy as np
from tqdm import tqdm
from scipy.spatial import distance

from train_protos import create_protos, dataset_generator, trim_embeddings
from train_protos_iterable import label2id, get_labelmap
from labelmaps import xaigenre_id2label, gtzan_id2label, nsynth_id2label

seed = 42


def find_closest_sample(
    data_dir: Path = None,
    metadata_file: Path = None,
    proto_file: Path = None,
    output_file: Path = None,
    n_protos_per_label: int = 1,
    batch_size: int = 32,
    seed: int = 42,
    val_test_ratio: float = 0.1,
    timestamps: int = 300,
    trim_mode: str = "middle",
    temp: float = 0.1,
    dataset: str = None,
):
    if dataset == "gtzan":
        label_map = gtzan_id2label
    elif dataset == "nsynth":
        label_map = nsynth_id2label
    elif dataset == "xai_genre":
        label_map = xaigenre_id2label
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print("creating dataset")
    ds = Dataset.from_generator(
        dataset_generator,
        writer_batch_size=100,
        gen_kwargs={
            "metadata_file": metadata_file,
            "data_dir": data_dir,
            "dataset": dataset,
            "seed": seed,
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

    ds = ds.map(
        label2id,
        batched=True,
        fn_kwargs={"labelmap": get_labelmap(dataset)},
    )

    ds = ds.with_format("numpy")

    print("creating train splits")
    ds = ds.train_test_split(test_size=val_test_ratio, seed=seed)
    ds_train = ds["train"]
    ds_tran_val = ds_train.train_test_split(test_size=val_test_ratio, seed=seed)
    ds_train = ds_tran_val["train"]
    ds_val = ds_tran_val["test"]

    time_dim = ds_val["feature"][0].shape[0]
    feat_dim = ds_val["feature"][0].shape[1]
    print(f"time_dim: {time_dim}, feat_dim: {feat_dim}")

    labels = list(label_map.values())
    labels.sort()

    n_labels = len(labels)
    n_protos = n_labels * n_protos_per_label
    print(f"n_labels: {n_labels}, n_protos: {n_protos}")

    protos = create_protos(
        ds_train,
        "proto-file",
        (time_dim, feat_dim),
        n_protos_per_label=n_protos_per_label,
        labels=labels,
        proto_file=proto_file,
    )

    samples = []
    filenames = []

    for i in tqdm(range(len(protos))):
        proto = protos[i].numpy().flatten()
        distances = [
            distance.cosine(sample.flatten(), proto) for sample in ds_train["feature"]
        ]
        closest = np.argmin(distances)

        samples.append(ds_train["feature"][closest])
        filenames.append(ds_train["filename"][closest])

    samples = np.array(samples)
    np.save(output_file, samples)

    with open(output_file.with_suffix(""), "w") as f:
        f.write("\n".join(filenames))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--proto-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)

    parser.add_argument("--n-protos-per-label", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--val-test-ratio", type=float, default=0.1)
    parser.add_argument("--timestamps", type=int, default=300)
    parser.add_argument("--trim-mode", type=str, default="middle")
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, choices=["gtzan", "nsynth", "xai_genre"])

    args = parser.parse_args()
    find_closest_sample(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        proto_file=args.proto_file,
        output_file=args.output_file,
        n_protos_per_label=args.n_protos_per_label,
        batch_size=args.batch_size,
        seed=args.seed,
        val_test_ratio=args.val_test_ratio,
        timestamps=args.timestamps,
        trim_mode=args.trim_mode,
        temp=args.temp,
        dataset=args.dataset,
    )
