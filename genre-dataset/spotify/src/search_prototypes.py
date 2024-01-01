import time
import yaml
import click
from search_by_query import main as search

FILE_PROTOTYPES = "../../prototypes-AM-selected-genres.yaml"
FILE_MATCHES = "../../prototypes-AM-selected-genres.yaml.spotifyapi.tsv"


def select_candidate(candidate_tracks):
    print("\n")
    print(f">>> Select a match for the track {artist} - {track}:")
    choices = []
    for i, candidate_track in enumerate(candidate_tracks):
        print(f'{i}. {" - ".join(candidate_track)}')
        choices.append(f'{i}')
    i_nomatch  = len(candidate_tracks)
    print(f'{i_nomatch}. NO MATCH')
    choices.append(f'{i_nomatch}')
    choice = int(click.prompt("Select:", type=click.Choice(choices), show_default=False))
    if choice == i_nomatch:
        return None
    else:
        return candidate_tracks[choice]


with open(FILE_PROTOTYPES, 'r') as f:
    prototypes_dict = yaml.safe_load(f)

with open(FILE_MATCHES, 'a') as f:
    for genre, tracks in prototypes_dict.items():
        genre = genre.lower()
        print(genre)
        for track in tracks:
            track, artist = track.split('---')
            print(f'Searching a match for {genre} - {artist} - {track}')

            query = f"artist:{artist} track:{track}"
            candidate_tracks = search(query, filter=None, limit=10, offset=0, wildcard=None, market=None)

            match = None
            # First, try to match automatically.
            for candidate_track in candidate_tracks:
                print(candidate_track[0], artist, candidate_track[1], track)
                if candidate_track[0].lower() == artist.lower() and candidate_track[1].lower() == track.lower():
                    print("Found exact match")
                    match = candidate_track

            # If a match was not found, ask user to select the match manually.
            if match is None:
                if len(candidate_tracks):
                    match = select_candidate(candidate_tracks)

            if match is None:
                f.write(f'{genre}\t{artist}\t{track}\tNO MATCH\n')
            else:
                f.write(f'{genre}\t{artist}\t{track}\t{match[2]}\t{match[0]}\t{match[1]}\n')
            time.sleep(3)
