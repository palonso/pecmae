import argparse
import math
from more_termcolor import colored
from pathlib import Path
import random
import sys
import time


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))

from utils.market import (
    available_markets,
    alpha2_code_to_name,
    get_alpha2_code_to_name_dict,
    remove_unavailable_markets,
)
from utils.spotify import (
    analysis_exists,
    get_spotify_client,
    search_query,
    preview_url_exists,
)
from utils.utils import write_yaml

OUTPUT_DIR = SCRIPT_DIR.parent / "data"
ALPHA2_DICT = get_alpha2_code_to_name_dict()
AVAILABLE_MARKETS = remove_unavailable_markets(available_markets)


def get_random_int(top: int, bottom: int = 0) -> int:
    return bottom + math.floor(random.uniform(0, 1) * top)


def get_random_market() -> str:
    return AVAILABLE_MARKETS[get_random_int(len(AVAILABLE_MARKETS))]


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
        chars += "!@%&+ç<>œøπΩßµ"  # give access to other content related with specific cultures like π and µ are related with Russian and Greek songs.

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


def get_random_id(
    client: object,
    year: str,
    market: str,
    display: bool = False,
) -> str:

    if market:
        query, offset, _ = get_random_search_args()
    else:
        query, offset, market = get_random_search_args()

    if year:
        query = f"{query} year:{year}"

    if display:
        print(
            colored(
                f"query: {query}\noffset: {offset}\nmarket: {market}-{alpha2_code_to_name(market, ALPHA2_DICT)}",
                "green",
            )
        )

    results = search_query(
        client, query, type="track", limit=1, offset=offset, market=market
    )

    # check results["tracks"]["items"] is not empty
    track_id = ""
    if "tracks" in results.keys():
        if results["tracks"]["items"]:
            track = results["tracks"]["items"][0]
            track_id = track["id"]
            if display:
                print(
                    f"{track['artists'][0]['name']} - {track['name']} - {track_id}"
                )

    return track_id


def get_unique_id(
    client: object,
    random_ids: dict,
    year: str = None,
    market: str = None,
    max_n_queries: int = 10,
) -> str:
    is_non_unique = True
    n = 0
    while is_non_unique and n < max_n_queries:
        random_id = get_random_id(client, year, market, display=True)
        # check uniqueness
        if random_id:
            is_non_unique = (
                False if random_id not in list(random_ids.values()) else True
            )
            if not is_non_unique:
                # get track and check preview_url is not None
                if not preview_url_exists(client, random_id):
                    is_non_unique = True
                    print(
                        colored(f"{random_id} doesn't provide preview", "red")
                    )

                # set is_non_unique to False if analysis is not provided
                if not analysis_exists(client, random_id):
                    is_non_unique = True
                    print(
                        colored(f"{random_id} doesn't provide analysis", "red")
                    )
        else:
            print(colored(f"random_id is empty.", "red"))
        n += 1
    return random_id


def main(
    n_spotify_ids: int,
    output_path: Path,
    year: str,
    market: str,
) -> None:

    # initialize spotify client
    sp = get_spotify_client()

    random_ids = {}
    start_time = time.time()
    for n in range(n_spotify_ids):
        random_id = get_unique_id(sp, random_ids, year, market)
        random_ids[n] = random_id
        print(colored(f"{n}: {random_id}", "cyan"))

    print(random_ids, f"\nExecution time: {time.time() - start_time:.3}[s]")

    # write yaml with spotify ids
    write_yaml(output_path, random_ids)


def handle_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Generates a list of N spotify ids based on random queries using wildcards, year and markets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--n_spotify_ids",
        help="The amount of spotify ids to generate.",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Directory to save yaml file with the output ids.",
        type=Path,
        required=False,
        default=OUTPUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.yaml",
    )
    parser.add_argument(
        "-y",
        "--year",
        help="Year of release date",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "-m",
        "--market",
        help=f"Skip random market.",
        type=str,
        required=False,
        default=None,
        choices=AVAILABLE_MARKETS,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = handle_args()
    main(**vars(args))
