from argparse import ArgumentParser
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, ASTModel
from essentia.standard import MonoLoader


parser = ArgumentParser()
parser.add_argument("audio_dir", type=Path)
parser.add_argument("embeddings_dir", type=Path)
parser.add_argument("--format", type=str, default="wav", choices=["wav", "mp3"])
parser.add_argument("--device", type=str, default="cuda:0")

args = parser.parse_args()

device = args.device

sample_rate = 16000
duration = 30
layer = 6

model_name = "mtg-upf/discogs-maest-30s-pw-129e"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_name, trust_remote_code=True
)
model = ASTModel.from_pretrained(model_name, output_hidden_states=True).to(device)

monoloader = MonoLoader(sampleRate=sample_rate)

glob_pattern = str(args.audio_dir / "**" / f"*.{args.format}")
audio_files = glob(glob_pattern, recursive=True)


min_frames = sample_rate * duration

for audio_file in tqdm(audio_files):
    audio_file = Path(audio_file)

    output_path = (
        args.embeddings_dir / audio_file.parent.name / audio_file.name
    ).with_suffix(".npy")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        continue

    monoloader.configure(
        filename=str(audio_file),
        sampleRate=sample_rate,
        resampleQuality=4,
    )
    xorig = monoloader()

    inputs = feature_extractor(
        xorig,
        return_tensors="pt",
        sampling_rate=sample_rate,
    ).to(device)

    with torch.no_grad():
        embeddings = model(**inputs).hidden_states[layer]

    embeddings = embeddings.cpu().numpy()
    embeddings = np.mean(embeddings, axis=0).squeeze()

    cls = embeddings[0]
    dist = embeddings[1]
    avg = np.mean(embeddings[2:], axis=0)

    embeddings = np.concatenate([cls, dist, avg])

    np.save(output_path, embeddings)
