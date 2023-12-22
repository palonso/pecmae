import csv
import json
import jsonlines
from pathlib import Path
import yaml


def read_yaml(yaml_fn: Path) -> dict:
    """Read yaml file and return the dictionary.

    Args:
        yaml_fn (Path): path to yaml file.

    Returns:
        dict: yaml data.
    """
    with open(str(yaml_fn), "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def write_yaml(output_fn: Path, data_dict: dict, sort_keys: bool = False) -> None:
    with open(str(output_fn), "w") as file:
        yaml.dump(data_dict, file, sort_keys=sort_keys)
    print(f"Saved yaml file in {output_fn}")


def read_csv(file_fn: Path) -> tuple:
    """Read csv file.

    Args:
        file_fn (Path): csv file path.

    Returns:
        Tuple: file and csv data.
    """
    file = open(str(file_fn))
    csvreader = csv.reader(file)
    return file, csvreader


def load_json(json_path: Path) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    return data


def write_json(file_fn: Path, data: dict) -> None:
    with open(file_fn, "w") as outfile:
        json.dump(data, outfile)


def write_jsonl(json_data, file_loc=False):
    """Write list object to newline-delimited JSON format.
    :param json_data: data in an object that is to be converted to JSONL format
    :type json_data: list
    :param file_loc: location of file to write to
    :type file_loc: str, bool
    """
    with jsonlines.open(file_loc, "w") as writer:
        writer.write_all(json_data)


def read_jsonl(json_path: Path) -> list:
    with jsonlines.open(str(json_path), "r") as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst
