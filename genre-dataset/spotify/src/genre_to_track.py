"""
This script changes the format of the YAML file generated with genre_ids.py
It generates a new YAML file from genre_ids.py (track_ids split by genre)
that only provides the track_ids.
This functionality should be included in genre_ids.py as a postprocessing
of the resultsing YAML file.
"""

import argparse
from more_termcolor import colored
from pathlib import Path
import sys
import time

SCRIPT_DIR = Path(__file__).parent.resolve()

from utils.utils import read_yaml, write_yaml

OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "generated"

def main(input_path: Path, output_dir: Path, data_dir: Path)->None:

    # read YAML file from genre_ids.py
    genre_ids = read_yaml(input_path)

    # count number of track_ids per genre at the input
    n = 0
    stats = {k:{} for k in genre_ids.keys()}
    ids_dict = {}
    annotation_dict = {}
    genre_dict = {k:[] for k in genre_ids.keys()}

    for key in stats.keys():

        stats[key]["n_track_ids"] = len(genre_ids[key])
        stats[key]["available"] = 0

        # count the availability of each genre in terms of track_ids
        for track_dict in genre_ids[key]:
            # filter all track_ids that don't provide audio and metadata
            if track_dict["preview"] and track_dict["analysis"]:
                stats[key]["available"] += 1
                # put all track_ids together and save it as all_track_ids.yaml
                ids_dict[n] = track_dict["id"]
                annotation_dict[n] = key
                # store filtered track_ids for each genre
                genre_dict[key].append(track_dict["id"])
                n += 1

    stats.update({"all": n})

    # save all these YAMLs files in a output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    write_yaml(output_dir / "ids.yaml", ids_dict)
    write_yaml(output_dir / "genres.yaml", annotation_dict)

    # save all genre dicts
    genre_dir = output_dir / "genres"
    genre_dir.mkdir(parents=True, exist_ok=True)
    for key in genre_dict.keys():
        write_yaml(genre_dir / f"{key}.yaml", genre_dict[key])

    # save statistics with the number of track ids at input and
    # the filtered track_ids, the same as output
    write_yaml(output_dir / "stats.yaml", stats)


def handle_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Generates a list of N spotify ids based on random queries using wildcards, year and markets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_path",
        help="Path to genre yaml file.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Directory to save yaml file with the output ids.",
        type=Path,
        required=True,
        default=OUTPUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.yaml",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        help="Data directory with annotations and previews to filter not available tracks.",
        type=Path,
        required=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = handle_args()
    main(**vars(args))
