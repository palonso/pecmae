import yaml
import argparse

import pandas as pd

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("proto_file", type=Path)
parser.add_argument("metadata_file", type=Path)

args = parser.parse_args()

with open(args.proto_file, "r") as f:
    protos = yaml.safe_load(f)

metadata = pd.read_csv(args.metadata_file, sep="\t", header=None)

out_data = dict()

for label, song_artist_list in protos.items():
    out_data[label] = []
    for song_artist in song_artist_list:
        song, artist = song_artist.split("---")
        track = metadata.loc[metadata[2] == song]

        sid = track[3].iloc[0]

        if sid != "NO MATCH":
            out_data[label].append(sid)

with open(str(args.proto_file) + ".ids", "w") as f:
    yaml.dump(out_data, f)
