# Flatten prototypes YAML metadata into TSV

import sys
import yaml

prototypes_dict = yaml.safe_load(sys.stdin)
for genre, tracks in prototypes_dict.items():
    genre = genre.lower()
    for track in tracks:
        track, artist = track.split('---')
        print('\t'.join((genre, artist, track)))
