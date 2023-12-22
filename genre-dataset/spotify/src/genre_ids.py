import argparse
import math
from more_termcolor import colored
from pathlib import Path
import random
import sys
import time


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))

from utils.market import available_markets
from utils.spotify import (
    analysis_exists,
    check_preview,
    dict_has_tracks,
    get_spotify_client,
    search_query,
)
from utils.utils import read_yaml, write_yaml, load_json

OUTPUT_DIR = SCRIPT_DIR.parent / "data"
SPOTIFY_GENRES_PATH = SCRIPT_DIR / "utils" / "genres.json"


def get_random_int(
    top: int, bottom: int = 0, min: int = 0, max: int = 1
) -> int:
    return bottom + math.floor(random.uniform(min, max) * top)


def get_random_market() -> str:
    return available_markets[get_random_int(len(available_markets))]


def get_random_offset(query_limit: int = 1) -> int:
    #! looks like offset+limit <= 1000 per query (so if max_limit=50,max_offset=950)
    spotify_limit = 1000  # query_limit + offset <= 1000
    max_offset = spotify_limit - query_limit
    random_offset = get_random_int(max_offset)
    return random_offset


def get_random_search(numbers: bool = True, special_chars: bool = True) -> str:

    # define all the possible wild cards
    chars = " abcdefghijklmnñopqrstuvwxyz"

    if numbers:
        chars += "012345670"

    if special_chars:
        chars += "!@%&+ç<>œ øπΩßµ"  # give access to other content related with specific cultures like π and µ are related with Russian and Greek songs.

    random_char = chars[get_random_int(len(chars))]

    # the search might changes if you add a %
    random_search = (
        random_char + "%" if round(random.uniform(0, 1)) else random_char
    )

    return random_search


def get_random_search_args() -> tuple:
    # make random wildcard generator using characters and numbers <char>%
    query = get_random_search()

    offset = get_random_offset()

    # random market
    market = get_random_market()
    return query, offset, market


def build_query(genre: str, years: list):
    if years:
        if len(years) == 2:
            year = get_random_int(top=years[0] - years[1], bottom=years[1])
        else:
            year = years[0]

    query = f"genre:'{genre}'"

    if year:
        query = f"{query} year:{year}"
    return query


def get_out_dict(track_id, preview, analysis) -> dict:
    return {
        "id": track_id,
        "preview": preview,
        "analysis": analysis,
    }


def display_genre_status(data, genre):
    print(colored(f"{genre} has {len(data[genre])} ids.", "green"))


def append_instance(
    data: dict,
    genre: str,
    track: object,
    preview: bool,
    analysis: bool,
    output_path: Path,
):
    data[genre].append(get_out_dict(track["id"], preview, analysis))
    display_genre_status(data, genre)

    if not output_path.parent.exists():
        output_path.parent.mkdir(exist_ok=True, parents=True)

    write_yaml(output_path, data)


def append_in_yaml(
    genre: str,
    track: object,
    preview: bool,
    analysis: bool,
    output_path: Path,
):
    # if yaml exists append track id otherwise just initialize a yaml file
    if output_path.exists():

        data = read_yaml(output_path)   # read yaml file with genre ids

        # if the genre already exists in yaml file, just append the song_id
        if genre in list(data.keys()):

            # add song_id if it is unique
            if track["id"] not in [
                data_track["id"] for data_track in data[genre]
            ]:
                append_instance(
                    data, genre, track, preview, analysis, output_path
                )

            else:
                print(
                    colored(f"{track['id']} already exists in {genre}", "red")
                )
        else:
            # we initialize a genre list when the genre doesn't exists in yaml file
            data[genre] = []
            append_instance(data, genre, track, preview, analysis, output_path)
    else:
        # initialize a dict to create the yaml file
        data = {genre: []}
        append_instance(data, genre, track, preview, analysis, output_path)


def update_empty_genres(random_ids: dict, genre: str, genres: list):
    print(colored(f"Add {genre} to empty genre list.", "magenta"))
    genres.remove(genre)
    if genre in list(random_ids.keys()):
        random_ids.pop(genre)


def update_full_genres(
    random_ids: dict, full_genres: list, genre: str, idx: int, genres: list
):
    # update full genres list
    full_genres.append(genre)
    random_ids.pop(genre)
    genres.remove(genre)
    print(colored(f"{genre} already has {idx} track ids.", "red"))


def update_genres(
    results: dict,
    random_ids: dict,
    genre: str,
    genres: list,
    idx: int,
    output_path: Path,
):
    # add genre to the random_ids dict if it doesn't exist
    if genre not in list(random_ids.keys()):
        random_ids[genre] = []

   # we get the amount of tracks for this query
   # and some initialization before searching
    length = len(results["tracks"]["items"])
    is_non_unique = True
    idxs = list(range(length))

    # we iterate over all the tracks to get a random one
    # which has a preview available
    while is_non_unique and len(idxs) > 0:

        random.shuffle(idxs)                    # shuffling the indexes
        random_idx = get_random_int(len(idxs))  # get a random index
        idxs.pop(random_idx)                    # remove idx already tested
        track = results["tracks"]["items"][random_idx]  # get the track

        # check uniqueness and preview availability
        checkers = [
            track["id"] not in random_ids[genre],
            check_preview(track),  # ensure preview is contained
        ]

        # if all is fine we append the track id to the yaml file
        if all(checkers):
            # TODO: try to donload data (preview, analysis)
            # TODO: define annotation path
            # TODO: define audio path
            # TODO: if there is not issues append to yaml and break loop otherwise, remove donwloaded data and continue
            is_non_unique = False
            random_ids[genre].append(track["id"])
            #! this check is not working fine it is always True when it is not
            preview = check_preview(track)
            analysis = True
            # analysis = analysis_exists(client, track["id"])   #! generates errors sometimes
            print(
                f"[{idx}/{len(genres)}] - {genre}: {track['id']} - {preview}"  # - {analysis}"
            )
            append_in_yaml(
                genre,
                track,
                preview,
                analysis,
                output_path,
            )


def get_query(client: object, genre: str, years: str, n: int) -> dict:
    query = build_query(genre, years)
    # In the first round use offset = 0 and market=None
    if n:
        wildcard, offset, market = get_random_search_args()
        query = f"{query} {wildcard}"
    else:
        offset = 0
        market = None

    try:
        results = search_query(
            client,
            query,
            type="track",
            limit=50,
            offset=offset,
            market=market,
        )
    except Exception as e:
        results = {"tracks": {"items": []}}
    return results, query


def fetch_genres(
    genres: list, client: object, n_tracks: int, years: list, output_path: Path
):

    # prepare some bins
    print(f"n_tracks: {n_tracks}")

    # if output_path exists, we load the yaml file as random ids and append more ids to it
    if output_path.exists():
        random_ids = {
            k: [value["id"] for value in v]
            for k, v in read_yaml(output_path).items()
        }
        #genres = list(random_ids.keys())
    else:
        random_ids = {genre: [] for genre in genres}

    empty_genres = list()
    full_genres = list()

    # iterate to get spotify ids for a number of tracks for each genre
    for n in range(n_tracks):

        print(colored(f"Round {n}", "bold"))
        random.shuffle(genres)      # shuffle genres at each iteration

        for m, genre in enumerate(genres):

            # make a random query with wildcards
            results, query = get_query(client, genre, years, n)

            # if there are tracks available in a genre, we will get one randomly
            # otherwise we will add the genre to empty_genres list
            # and remove it from genres list

            if dict_has_tracks(results):

                # check if the genre is not full (has less than n_tracks)
                if len(random_ids[genre]) < n_tracks:
                    update_genres(
                        results,
                        random_ids,
                        genre,
                        genres,
                        m,
                        output_path,
                    )
                    # TODO: for the next version, check if preview and analysis is available.
                    # TODO: if they are download it in genre folder with spotify id name
                    # TODO: think how to store the previews and annotations
                    # TODO: if there is any issue downloading data, just remove all data downloaded and don't update genre
                    # TODO: if they are not, just continue
                    # TODO: it would be nice to detect those genres with some difficulties based on lenght of randomids
                    # TODO: (if len(random_ids[genre] < n*4/5)) remove special characters,  (if len(random_ids[genre] < 3n/4)) remove numbers too (if len(random_ids[genre] < n/2)) remove consonants too
                    # TODO: in those cases, we can modify the wildcards using the most common characters like vowels and avoid numbers and consonants
                else:
                    # update full genres list
                    update_full_genres(
                        random_ids, full_genres, genre, n, genres
                    )
            elif not dict_has_tracks(results) and n == 0:
                #  append it to empty_genres and remove it from genres list
                update_empty_genres(random_ids, genre, genres)
            else:
                print(f"Empty results in query: {query}")

    print(f"empty genres: {empty_genres}")
    print(f"full_genres: {full_genres}")

    # save empty_genres in a dict to be reused for any fecth process
    write_yaml(output_path.parent / "empty_genres.yaml", empty_genres)
    write_yaml(output_path.parent / "full_genres.yaml", full_genres)


def get_genre_ids(
    client: object,
    limit: int,
    genre: str,
    year: str = None,
    offset: int = 0,
    market: str = None,
    display: bool = True,
) -> list:

    query = f"genre:'{genre}'"

    if year:
        query = f"{query} year:{year}"

    if display:
        print(
            colored(
                f"query: {query}\noffset: {offset}\nmarket: {market}",
                "green",
            )
        )

    results = search_query(
        client,
        query,
        type="track",
        limit=limit,
        offset=offset,
        market=market,
    )
    if results["tracks"]["total"]:
        # TODO: get the length of tracks
        # TODO: if length of tracks == 50 make more queries in this genre
        for n, track in enumerate(results["tracks"]["items"]):
            track_id = track["id"]
            if display:
                print(
                    f"{n}: {track['artists'][0]['name']} - {track['name']} - {track_id} - {track['preview_url'] != None} - {analysis_exists(client, track['id'])}"
                )
        # TODO: if len(results) > n_spotify_ids stop the process and return n random ids
        output = {genre: [track["id"] for track in results["tracks"]["items"]]}
    else:
        output = {genre: []}
    return output


def analyze_genre(
    client: object,
    genre: str,
    limit: int = 50,
    offset: int = 0,
    year: int = None,
):
    total_genre_ids = 0
    is_completed = False
    while not is_completed:
        genre_ids = get_genre_ids(
            client, limit, genre, offset=offset, year=year, display=True
        )
        length = len(genre_ids[genre])
        if length == limit and limit + offset < 1000:
            offset += limit
            total_genre_ids += limit
        else:
            #! there are two cases, genre with more than 1k tracks and genre with no more tracks
            #! if offset = 1000 - limit then we can get more tracks iterating for all the possible wild cards and applying the same strategy with limit and offset
            total_genre_ids += length
            is_completed = True
    return total_genre_ids


def analyze_genres(genres: list, client: object, year: int, output_path: Path):
    results = {}
    start_time = time.time()
    for n, genre in enumerate(genres):
        # analyse track ids per genre
        total_genre_ids = analyze_genre(client, genre, year)
        results[n] = {genre: total_genre_ids}
        print(f"{n}: {genre}: {total_genre_ids}")

    print(results, f"\nExecution time:{time.time() - start_time}[s]")

    # write yaml with spotify ids
    write_yaml(output_path, results)


def is_spotify_genre(genres: list, spotify_genres: list) -> bool:
    return set(genres).issubset(spotify_genres)


def main(
    analyze: bool,
    n_spotify_ids: int,
    output_path: Path,
    genres: list,
    genres_path: Path,
    min_year: str,
    max_year: str,
    force: bool,
) -> None:

    # if force remove outpath if it alreaady exist in fetch genres
    if force and output_path.exists():
        output_path.unlink()

    # initialize spotify client
    sp = get_spotify_client()

    # replace everynoise genre list with an external one provided by command line
    # if input_genres_path:
    #     if input_genres_path.exists() and input_genres_path.is_file():
    #         GENRES_PATH = input_genres_path

    # load genre list from everynoise.com available in utils/genres.json
    spotify_genres = load_json(SPOTIFY_GENRES_PATH)

    # load external genre file to fetch tracks for specific genres, it replace genre list
    if genres_path:
        genres = load_json(genres_path)

    #! this fails a lot when spotify API return some track ids for those genres
    # we check genres when they are externally provided in command line
    # if genres:
    #     if not is_spotify_genre(genres, spotify_genres):
    #         for genre in genres:
    #             if set([genre]).issubset(spotify_genres):
    #                 raise ValueError(
    #                     f"error: {genre} is not available in Spotify genre list."
    #                 )
    # else:
    #     genres = spotify_genres

    if not genres:
        genres = spotify_genres

    years = list(map(int, [max_year, min_year]))

    # analyze or gather spotify ids for each genre
    if analyze:
        analyze_genres(genres, sp, years, output_path)
    else:
        fetch_genres(genres, sp, n_spotify_ids, years, output_path)


def handle_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Analyze and fetch spotify ids for an specific genre list to generate datasets based on genres.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-a",
        "--analyze",
        help="Analyze number of tracks per genre.",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-n",
        "--n_spotify_ids",
        help="The amount of spotify ids per genre.",
        type=int,
        required=False,
        default=50,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Directory to save yaml file with the output ids.",
        type=Path,
        required=False,
        default=OUTPUT_DIR / "genres_out.yaml",
    )
    parser.add_argument(
        "-g",
        "--genres",
        nargs="+",
        help=f"Genres to use as genre list. Replace everynoise.com genre list.",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-gp",
        "--genres_path",
        help="JSON file with a list of genres. By default everynoise genres.",
        type=Path,
        required=False,
        default=SCRIPT_DIR / "utils" / "genres.json",
    )
    parser.add_argument(
        "--min-year",
        help="Minimum year of release date",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--max-year",
        help="Maximum year of release date or year of release when minimum year is not provided.",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Force clean featching.",
        action="store_true",
        required=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = handle_args()
    main(**vars(args))
