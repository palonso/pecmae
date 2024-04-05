import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from essentia.standard import MonoLoader, MonoWriter


def randomize_audio(audio, grain_size: float = 0.05, sr: int = 44100):
    grain_size_samples = int(sr * grain_size)
    cut = len(audio) % grain_size_samples

    audio = audio[:-cut].reshape(-1, grain_size_samples)

    perm = np.arange(audio.shape[0])
    np.random.shuffle(perm)

    audio = audio[perm]

    return audio.flatten()


parser = argparse.ArgumentParser()
parser.add_argument("metadata_file", type=Path)
parser.add_argument("input_dir", type=Path)
parser.add_argument("output_dir", type=Path)
parser.add_argument(
    "--dataset", choices=["gtzan", "nsynth", "xai_genre", "medley_solos"], required=True
)
parser.add_argument("--grain-size", type=float, default=0.05)

args = parser.parse_args()

dataset = args.dataset
grain_size = args.grain_size

output_dir = args.output_dir / f"{grain_size}ms_grains" / dataset


metadata = pd.read_csv(args.metadata_file, sep="\t", header=None)


sr = 22050

for row in tqdm(metadata.itertuples(), total=len(metadata)):
    genre = row[1]
    filename = row[2]

    if dataset == "gtzan":
        path = args.input_dir / f"{filename[:-4]}.wav"
    elif dataset == "nsynth":
        path = args.input_dir / genre
    elif dataset == "xai_genre":
        path = args.input_dir / genre / "audio" / f"{filename}.mp3"
    elif dataset == "medley_solos":
        path = args.input_dir / f"{filename[:-4]}.wav"

    output_path = output_dir / path.relative_to(args.input_dir)
    print(output_path)

    if output_path.exists():
        print("skipping")
        continue

    audio = MonoLoader(filename=str(path), sampleRate=sr, resampleQuality=4)()

    audio = randomize_audio(audio, sr=sr, grain_size=grain_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    MonoWriter(filename=str(output_path))(audio)
