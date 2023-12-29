import argparse
from more_termcolor import colored
from pathlib import Path
import sys
from time import sleep

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))

from utils.market import (
    available_markets,
    get_alpha2_code_to_name_dict,
    remove_unavailable_markets,
)
from utils.spotify import (
    get_spotify_client,
    search_query,
    analysis_exists,
    check_preview,
)

AVAILABLE_MARKETS = remove_unavailable_markets(available_markets)


def main(query: str, filter: str, limit: int, offset: int, wildcard: str, market: str):
    """
    Search by item in Spotyfy by query.

    Args:
        query (str): Query to search
        filter (str): Specify the filter. Choices in get_available_query_filters_list()
        limit (int): The limit of items (tracks). 50 as maximum.
        offset (int): The limit of items (tracks). limit + offset <= 1000
        wildcard (str): Provide tracks that include wildcard. Usually defined as <character>%
        market(str): Indicates the location to search. Useful to get access to unpopular music.
    """

    # initialize spotify client
    sp = get_spotify_client()

    # build query with filter and wildcard
    if filter:
        query = f"{filter}:{query}"

    # with wildcards, it returns the same result for <char> than %<char>, it changes for <char>%
    if wildcard:
        query = f"{wildcard} {query}"

    print(colored(f"query: {query}", "yellow"))

    markets = AVAILABLE_MARKETS if market == "all" else [market]
    # get countries names
    countries_dict = get_alpha2_code_to_name_dict()

    def single_market_query(market):
        print("Executing a query!")
        results = search_query(sp, query, limit=limit, offset=offset, market=market)
        print("Got some results!")
        if results["tracks"]["total"]:
            if market:
                print(colored(f"At {countries_dict[market]} ({market}) market", "cyan"))
            for n, track in enumerate(results["tracks"]["items"]):
                if check_preview(track): #and analysis_exists(sp, track['id']):
                    print(f"{query}\t{track['id']}\t{track['artists'][0]['name']}\t{track['name']}")
                #print(
                #    f"{n}: {track['artists'][0]['name']} - {track['name']} - {track['id']}"
                #    f" - preview: {check_preview(track)} - analysis: {analysis_exists(sp, track['id'])}",
                #)
                #sleep(10)
        else:
            print(f"{market}: {results['tracks']['total']} songs")

    if markets:
        for market in markets:
            single_market_query(market)
    else:
        single_market_query(market)


def get_available_query_filters_list():
    return [
        "album",
        "artist",
        "track",
        "year",
        "upc",
        "tag:hipster",
        "tag:new",
        "isrc",
        "genre",
    ]


def handle_args():
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description="This script let you run quereis to Spotify API for testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Query to search. It serves for name, artist, track, genre, anything you want to search.",
        type=str,
        required=False,
        default="",
    )
    parser.add_argument(
        "-f",
        "--filter",
        help="Narrow down your search using field filters.",
        type=str,
        required=False,
        default=None,
        choices=get_available_query_filters_list(),
    )
    parser.add_argument(
        "-l",
        "--limit",
        help="List of keys to extract in separated annotation files.",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "-o",
        "--offset",
        help="The index of the first item to return. Very valuable to randomize.",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "-w",
        "--wildcard",
        help="A wildcard character used to search.",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-m",
        "--market",
        help="An ISO 3166-1 alpha-2 country code. Kind of the country the user is from",
        type=str,
        required=False,
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = handle_args()
    main(**vars(args))
