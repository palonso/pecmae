from argparse import ArgumentParser
from pathlib import Path
from distutils.dir_util import copy_tree
from shutil import rmtree

from datasets import load_dataset


script_dir = Path(__file__).resolve().parent

datasets = {
    "gtzan": "marsyas/gtzan",
    # "medley_solos_db": medley_solos_db,
}


def download(dataset_name: str, data_home: Path):
    """Download dataset."""

    # Make path relative to the script dir
    dataset = load_dataset(
        datasets[dataset_name],
    )

    # ugly way to get the dataset path
    audio_path = dataset["train"][0]["audio"]["path"]
    dataset_path = Path(audio_path).parent.parent

    data_home = script_dir / data_home / dataset_name
    copy_tree(str(dataset_path), str(data_home))

    # clean cache
    rmtree(str(dataset_path))


if __name__ == "__main__":
    parser = ArgumentParser(description="Download datasets used in PECMAC experiments.")

    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset to be downloaded.",
        choices=["gtzan", "medley_solos_db"],
    )
    parser.add_argument(
        "--data-home",
        type=Path,
        help="Path where the dataset will be stored.",
        default="../audio/",
    )

    args = parser.parse_args()

    download(args.dataset_name, args.data_home)
