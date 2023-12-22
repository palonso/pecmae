"""
This script search preview in all Spotify market availables.
"""
import argparse
from more_termcolor import colored
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))

from utils.market import (
    alpha2_code_to_name,
    get_alpha2_code_to_name_dict,
    remove_unavailable_markets,
)
from utils.spotify import get_spotify_client, search_query, check_preview


def main(spotify_id: str):

    # initialize spotify client
    sp = get_spotify_client()

    track = sp.track(spotify_id)

    # get artist and song name
    artist = " & ".join([artist["name"] for artist in track["artists"]])
    title = track["name"]

    # remove KR (Republique of Korea) and XK markets, they have not previews available
    available_markets = track["available_markets"]
    remove_unavailable_markets(available_markets)
    n_markets = len(available_markets)

    # build query
    query = f"{artist} {title}"
    print(f"query: {query}")

    # get countries names
    countries_dict = get_alpha2_code_to_name_dict()

    # iterate for each market and provide market and country name where the preview is available.
    preview_markets = []
    for n, market in enumerate(available_markets):
        # with artist and song name to check if preview is available for an specific market
        result = search_query(sp, query, limit=1, market=market)
        if result["tracks"]["total"]:
            is_preview_available = check_preview(result["tracks"]["items"][0])
            message = f"[{n}/{n_markets}] - {market}/{alpha2_code_to_name(market, countries_dict)}: {is_preview_available}"
            color = "red"
            if is_preview_available:
                preview_markets.append(market)
                color = "green"
            print(colored(message, color))
        else:
            print(
                f"[{n}/{n_markets}] - not track available in {market}/{alpha2_code_to_name(market, countries_dict)}"
            )
    print(
        colored(
            f"Preview is available in {len(preview_markets)} markets: {preview_markets}.",
            "yellow",
        )
    )


def handle_args() -> dict:
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description="This script checks the preview availability of an spotify id at all markets. \
                     It lets analyze if previews are available for all markets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--spotify_id",
        help="Spotify ID to check if preview is available at some market.",
        type=str,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = handle_args()
    main(**vars(args))
