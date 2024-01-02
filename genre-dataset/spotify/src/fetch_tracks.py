import argparse
from more_termcolor import colored
from pathlib import Path
import shutil
import time
import wget


SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "out"

TEST_TRACK_IDS = ["2QrS0PaeXeeGCCLABYoQgE", "0t0F3Qyt3kKFU7rGxKa02t"]
TEST_ARTIST_IDS = [
    "spotify:artist:1AMMMSq3rJdZtFGnBXEkz7",
    "36QJpDe2go2KgaRleHCDTp",
]

# Sleep time in seconds to avoid API rate limiting.
SLEEP_TIME = 5


from utils.utils import read_yaml, write_yaml
from utils.spotify import fetch_analysis, fetch_features, get_spotify_client


def percentage(value: float, n: int, precision: int = 3) -> float:
    return round(100 * value / n, precision)


def display_reports(
    n_ids: int, existing: int, successed: int, failed: int
) -> None:
    print(f"\texisting: {existing}/{n_ids} ({percentage(existing, n_ids)}%)")
    print(f"\tcomputed: {successed}/{n_ids} ({percentage(successed, n_ids)}%)")
    print(f"\tfailed: {failed}/{n_ids} ({percentage(failed, n_ids)}%)")


def download_preview(url: str, out_path: Path, bar: bool = False) -> None:
    bar = wget.bar_adaptive if bar else bar
    success = True
    try:
        wget.download(url, out=str(out_path), bar=bar)
        print(f"Downloaded mp3 file in {out_path}")
    except Exception as e:
        print(e)
        success = False
    time.sleep(SLEEP_TIME)
    return success


def get_data_dict(track: dict, features: dict, analysis: dict) -> dict:
    return {
        "track": track,
        "audio_features": features,
        "audio_analysis": analysis,
    }


def display_features(features: dict, analysis: dict) -> None:
    print(colored(f"audio_features: {features}\n", "blue"))
    print(colored(f"audio_analysis: {analysis.keys()}", "red"))


def get_dir_with_two_char_id_folder(output_dir: Path, track_id: str) -> Path:
    output_dir = output_dir / track_id[:2]
    generate_dir(output_dir)
    return output_dir


def fetch_data(
    client: object, track: dict, out_fn: Path, is_test: bool = False
) -> dict:
    features = fetch_features(client, track["id"])[0]  # global features?
    analysis = fetch_analysis(client, track["id"])  # local features?

    if not any([features is None, analysis is None]):
        success = True if all([len(features), len(analysis)]) else False
    else:
        success = False

    # filter some data here
    # pop_keys = ["type", "id", "uri", "track_href", "analysis_url"]
    # [features.pop(key) for key in pop_keys]

    if is_test:
        display_features(features, analysis)

    data_dict = get_data_dict(track, features, analysis)

    # write annotations yaml file
    write_yaml(out_fn, data_dict)
    return success


def display_metadata(track: dict) -> None:
    print(
        colored(
            f"artist       : {'_&_'.join([artist['name'] for artist in track['artists']])}",
            "green",
        )
    )
    print(colored(f"track        : {track['name']}", "yellow"))
    print(colored(f"audio preview: {track['preview_url']}", "yellow"))
    # track id
    print(colored(f"spotify_id: {track['id']}", "cyan"))


def get_track_stem(track: dict, separator: str = ":") -> str:
    return f"{track['artists'][0]['name']}{separator}{track['name']}"


def track_info(counter: str, track: dict, is_test: bool = False) -> None:
    print(
        f"[{counter}] - Fetching data at {track['id']}:{get_track_stem(track)}"
    )
    if is_test:
        display_metadata(track)


def generate_dir(path: str) -> None:
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def create_output_dirs(base_out_dir: Path):
    return generate_dir(base_out_dir / "audio"), generate_dir(
        base_out_dir / "annotations"
    )


def prepare(
    output_dir: Path,
    force: bool,
    annotation_subfolder: str = None,
    audio_subfolder: str = None,
) -> tuple:
    # initialize spotify client
    client = get_spotify_client()

    # clean output directory
    if force and output_dir.exists():
        try:
            shutil.rmtree(output_dir)
        except OSError as e:
            print(f"Error: {output_dir} : {e.strerror}")

    # prepare output dirs
    annotation_dir = (
        output_dir / annotation_subfolder
        if annotation_subfolder
        else generate_dir(output_dir / "annotations")
    )
    audio_dir = (
        output_dir / audio_subfolder
        if audio_subfolder
        else generate_dir(output_dir / "audio")
    )
    return client, audio_dir, annotation_dir


def main(
    track_ids: list,
    output_dir: Path,
    force: bool,
    skip_annotations: bool,
    skip_preview: bool,
    input_yaml: Path = None,
    use_id_as_name: bool = False,
    two_char_ids_folder: bool = False,
    annotation_subfolder: str = None,
    audio_subfolder: str = None,
    offset: int = 0,
) -> tuple:

    # check if trac_ids or input_yaml is on
    if all([track_ids == None, input_yaml == None]):
        raise ValueError("Please provide some id as track_ids or a yaml file.")

    client, base_audio_dir, base_annotation_dir = prepare(
        output_dir, force, annotation_subfolder, audio_subfolder
    )

    #print(f"audio_dir: {base_audio_dir}")
    #print(f"annotation_dir: {base_annotation_dir}")

    if input_yaml:
        yaml_data = read_yaml(input_yaml)
        track_ids = list(yaml_data.values())

    n_track_ids = len(track_ids)

    start_time = time.time()
    successed, failed, existing = ({"audio": 0, "data": 0} for i in range(3))

    for n in range(offset, n_track_ids):
        try:
            download_meta = not skip_annotations
            download_preview = not skip_preview
            filename = track_ids[n] if use_id_as_name else f"{n}"
            if download_meta:
                if two_char_ids_folder:
                    annotation_dir = get_dir_with_two_char_id_folder(
                        base_annotation_dir, track_ids[n]
                    )
                else:
                    annotation_dir = base_annotation_dir
                annotation_path = annotation_dir / f"{filename}.yaml"
                if annotation_path.is_file() and annotation_path.exists():
                    print(
                        f"Skip downloading: analysis already exists {annotation_path}"
                    )
                    existing["data"] += 1
                    download_meta = False

            if download_preview:
                if two_char_ids_folder:
                    audio_dir = get_dir_with_two_char_id_folder(
                        base_audio_dir, track_ids[n]
                    )
                else:
                    audio_dir = base_audio_dir
                audio_path = audio_dir / f"{filename}.mp3"
                if audio_path.is_file() and audio_path.exists():
                    print(
                        f"Skip downloading: preview already exists in {audio_path}"
                    )
                    existing["audio"] += 1
                    download_preview = False

            if download_meta or download_preview:
                track = client.track(track_ids[n])
                track_info(f"{n}/{n_track_ids}", track)

                # Skip both preview and metadata if preview_url is null
                if track['preview_url']:
                    if download_meta:
                        analysis_success = fetch_data(client, track, annotation_path)
                        if analysis_success:
                            successed["data"] += 1
                        else:
                            failed["data"] += 1

                    if download_preview:
                        audio_success = download_preview(track["preview_url"], audio_path)
                        if audio_success:
                            successed["audio"] += 1
                        else:
                            failed["audio"] += 1
                else:
                    print(
                        colored(
                            f"Skip track: {track_ids[n]} has not audio preview available.",
                            "red",
                        )
                    )
                    failed["audio"] += 1
                    failed["data"] += 1

        except TimeoutError as te:
            print(f"TimeoutError in {track_ids[n]}: {te}")

        # TODO: provide a report with how many tracks were fetch and how much have annotation and how much have audio
    #print(f"Execution time: {round(time.time() - start_time, 2)}[s]")

    #print(f"Previews report:")
    #display_reports(
    #    n_track_ids, existing["audio"], successed["audio"], failed["audio"]
    #)

    #print(f"Analysis report:")
    #display_reports(
    #    n_track_ids, existing["data"], successed["data"], failed["data"]
    #)
    return existing, successed, failed


def handle_args():
    parser = argparse.ArgumentParser(
        description="Search spotify id with different input parameters (market, offset, genre, wildcard and query).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--track_ids",
        nargs="+",
        help=f"Spotify ids from tracks, i.e: {TEST_TRACK_IDS[0]}.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Directory to save data (mp3 & yaml).",
        type=Path,
        required=False,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Force and clean output directory.",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-a",
        "--skip_annotations",
        help="Skip downloading annotations (yaml).",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-p",
        "--skip_preview",
        help="Skip downloading audio preview (mp3).",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-i",
        "--input_yaml",
        help="Path to a YAMl file with spotify ids.",
        type=Path,
        required=False,
    )
    parser.add_argument(
        "-u",
        "--use_id_as_name",
        help="Use track id as output file name.",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-as",
        "--annotation_subfolder",
        help="Specify the name of annotation subfolder.",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ads",
        "--audio_subfolder",
        help="Specify the name of audio subfolder.",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-w",
        "--two_char_ids_folder",
        help="Create subfolder with the first two chars in spotify id.",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--offset",
        help="Index to start to fetch tracks.",
        type=int,
        required=False,
        default=0,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = handle_args()
    main(**vars(args))
