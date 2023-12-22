from more_termcolor import colored
from spotipy import Spotify
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials


def get_spotify_client(client_id: str = None, client_secret: str = None):
    return Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        )
    )


def search_query(
    client: Spotify,
    query: str,
    type: str = "track",
    limit: int = 10,
    offset: int = 0,
    market=None,
):
    try:
        #! seems max limit == 50, filter and show a warning value at the end
        results = client.search(
            query, type=type, limit=limit, offset=offset, market=market
        )
    except SpotifyException as e:
        print(colored(f"error in query: {query} : {e}", "red"))
        results = {}

    return results


def fetch_analysis(client: Spotify, track_id: str) -> dict:
    try:
        analysis = client.audio_analysis(track_id)
    except SpotifyException as e:
        print(colored(f"Error: {track_id} : {e}", "red"))
        analysis = {}
    return analysis


def fetch_features(client: Spotify, track_id: str) -> dict:
    try:
        features = client.audio_features(track_id)
    except SpotifyException as e:
        print(colored(f"Error: {track_id} : {e}", "red"))
        features = {}
    return features


def get_artist_top_tracks(client: object, artist_id: str) -> dict:
    results = client.artist_top_tracks(artist_id)
    return results["tracks"][:1]


def get_tracks(client: object, track_ids: list) -> list:
    return [client.track(track_id) for track_id in track_ids]


def analysis_exists(client: object, track_id: str) -> bool:
    return bool(fetch_analysis(client, track_id))


def preview_url_exists(client: object, track_id: str) -> bool:
    track = client.track(track_id)
    return check_preview(track)


def check_preview(track: object) -> bool:
    #! url might be available but preview data not though
    return (
        track["preview_url"] != None
        if "preview_url" in list(track.keys())
        else False
    )


def dict_has_tracks(results: dict) -> bool:
    if "tracks" in list(results.keys()):
        return bool(len(results["tracks"]["items"]))
    else:
        return False
