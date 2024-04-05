import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--metadata-file", type=Path, required=True)
parser.add_argument("--test-ratio", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

metadata_file = args.metadata_file
test_ratio = args.test_ratio
seed = args.seed

data = pd.read_csv(metadata_file, sep="\t", header=None)

print(f"data length: {len(data)}")
print("dropping duplicates...")
data = data.drop_duplicates(subset=1)
print(f"data length: {len(data)}")

base_dir = Path("feats_xai_genre_v1_diff/large_diffusion/audio/")

data = data[data[1].apply(lambda x: (base_dir / x).with_suffix(".npy").exists())]
print(f"data length: {len(data)}")

genre = data[0]
path = data[1]


genre = genre.apply(lambda x: x[6:])

genre_train, genre_test, path_train, path_test = train_test_split(
    genre,
    path,
    test_size=test_ratio,
    random_state=seed,
    stratify=genre,
)

genre_train, genre_val, path_train, path_val = train_test_split(
    genre_train,
    path_train,
    test_size=test_ratio,
    random_state=seed,
    stratify=genre_train,
)


ds_train = pd.DataFrame({"genre": genre_train, "path": path_train})
ds_val = pd.DataFrame({"genre": genre_val, "path": path_val})
ds_test = pd.DataFrame({"genre": genre_test, "path": path_test})

ds_train.to_csv(
    metadata_file.parent / f"{metadata_file.name}.train",
    sep="\t",
    index=False,
    header=False,
)
ds_val.to_csv(
    metadata_file.parent / f"{metadata_file.name}.val",
    sep="\t",
    index=False,
    header=False,
)
ds_test.to_csv(
    metadata_file.parent / f"{metadata_file.name}.test",
    sep="\t",
    index=False,
    header=False,
)
