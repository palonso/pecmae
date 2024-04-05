import json
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as L
import numpy as np
from torch import utils

from train_protos import ZinemaNet, create_protos, get_labelmap
from train_protos_splits import get_dataset


checkpoint_map = {
    "gtzan": {
        "3_protos": "tb_logs/zinemanet/version_506/checkpoints/epoch=5655-step=124432.ckpt",
        "5_protos": "tb_logs/zinemanet/version_509/checkpoints/epoch=4665-step=102652.ckpt",
        "10_protos": "tb_logs/zinemanet/version_512/checkpoints/epoch=3694-step=81290.ckpt",
        "20_protos": "tb_logs/zinemanet/version_503/checkpoints/epoch=2037-step=44836.ckpt",
    },
    "nsynth": {
        "3_protos": "",
        "5_protos": "",
        "10_protos": "",
        "20_protos": "",
    },
    "xai_genre": {
        "3_protos": "tb_logs/zinemanet/version_507/checkpoints/epoch=372-step=145843.ckpt",
        "5_protos": "tb_logs/zinemanet/version_510/checkpoints/epoch=379-step=148580.ckpt",
        "10_protos": "tb_logs/zinemanet/version_513/checkpoints/epoch=370-step=145061.ckpt",
        "20_protos": "tb_logs/zinemanet/version_504/checkpoints/epoch=377-step=147798.ckpt",
    },
    "medley_solos": {
        "3_protos": "",
        "5_protos": "",
        "10_protos": "",
        "20_protos": "tb_logs/zinemanet/version_502/checkpoints/epoch=2081-step=47886.ckpt",
    },
}


def test(
    data_dir: Path = None,
    data_dir_test: Path = None,
    metadata_file_test: Path = None,
    n_protos_per_label: int = 1,
    batch_size: int = 32,
    seed: int = 42,
    timestamps: int = 300,
    trim_mode: str = "middle",
    gpu_id: int = 0,
    alpha: float = 0.5,
    proto_loss: str = "l2",
    proto_loss_samples: str = "class",
    use_discriminator: bool = False,
    discriminator_type: str = "mlp",
    checkpoint: Path = None,
    dataset: str = None,
    time_summarization: str = "none",
    do_normalization: bool = False,
):
    hyperparams = locals()

    if data_dir_test is None:
        data_dir_test = data_dir

    with open("datasets_stats.json", "r") as f:
        stats = json.load(f)
    ds_mean = stats[dataset]["mean"]
    ds_std = stats[dataset]["std"]

    ds_test, _, _ = get_dataset(
        metadata_file_test,
        data_dir_test,
        dataset,
        timestamps,
        trim_mode,
        do_normalization=do_normalization,
        ds_mean=np.float32(ds_mean),
        ds_std=np.float32(ds_std),
    )

    if not checkpoint:
        checkpoint = checkpoint_map[dataset][f"{n_protos_per_label}_protos"]

    time_dim = timestamps
    feat_dim = ds_test[0]["feature"].shape[-1]
    print(f"time_dim: {time_dim}, feat_dim: {feat_dim}")

    labels = list(get_labelmap(dataset).keys())
    labels.sort()
    n_labels = len(labels)
    n_protos = n_labels * n_protos_per_label
    print(f"n_labels: {n_labels}, n_protos: {n_protos}")

    protos = create_protos(
        ds_test,
        "random",
        (time_dim, feat_dim),
        n_protos_per_label=n_protos_per_label,
        labels=list(range(n_labels)),
        proto_file=None,
        data_dir=None,
    )

    loader_test = utils.data.DataLoader(ds_test, batch_size=batch_size)

    model = ZinemaNet.load_from_checkpoint(
        checkpoint,
        protos=protos,
        time_dim=time_dim,
        feat_dim=feat_dim,
        n_labels=n_labels,
        batch_size=batch_size,
        alpha=alpha,
        proto_loss=proto_loss,
        proto_loss_samples=proto_loss_samples,
        use_discriminator=use_discriminator,
        discriminator_type=discriminator_type,
        time_summarization=time_summarization,
        do_normalization=do_normalization,
        ds_mean=ds_mean,
        ds_std=ds_std,
        labels=labels,
    )

    trainer = L.Trainer(devices=[gpu_id], precision="16-mixed")

    trainer.test(model, loader_test)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-dir-test", type=Path, required=False)
    parser.add_argument("--metadata-file-test", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "gtzan",
            "nsynth",
            "xai_genre",
            "medley_solos",
        ],
        required=True,
    )

    parser.add_argument("--n-protos-per-label", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestamps", type=int, default=300)
    parser.add_argument("--trim-mode", type=str, default="middle")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--proto-loss", default="l2", choices=["l1", "l2", "info_nce"])
    parser.add_argument("--proto-loss-samples", default="all", choices=["all", "class"])
    parser.add_argument(
        "--use-discriminator", type=lambda x: x == "True", default=False
    )
    parser.add_argument(
        "--discriminator-type", default="mlp", choices=["mlp", "conv", "linear"]
    )
    parser.add_argument(
        "--time-summarization",
        type=str,
        default="none",
        choices=["none", "lstm", "transformer", "dense_res"],
    )
    parser.add_argument("--do-normalization", type=lambda x: x == "True", default=False)

    args = parser.parse_args()
    test(
        data_dir=args.data_dir,
        data_dir_test=args.data_dir_test,
        metadata_file_test=args.metadata_file_test,
        n_protos_per_label=args.n_protos_per_label,
        batch_size=args.batch_size,
        seed=args.seed,
        timestamps=args.timestamps,
        trim_mode=args.trim_mode,
        gpu_id=args.gpu_id,
        alpha=args.alpha,
        proto_loss_samples=args.proto_loss_samples,
        proto_loss=args.proto_loss,
        use_discriminator=args.use_discriminator,
        discriminator_type=args.discriminator_type,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        time_summarization=args.time_summarization,
        do_normalization=args.do_normalization,
    )
