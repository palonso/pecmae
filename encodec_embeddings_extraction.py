from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor
from tqdm import tqdm
import numpy as np

from encodecmae import load_model

device = "cuda:0"

parser = ArgumentParser()

parser.add_argument("audio_dir", type=Path)
parser.add_argument("output_dir", type=Path)

args = parser.parse_args()
audio_dir = args.audio_dir
output_dir = args.output_dir


# audio_dir = Path("/mnt/mtgdb-audio/stable/genre_tzanetakis/audio/22kmono/")
dataset = load_dataset("audiofolder", data_dir=audio_dir, split="train")


# load the model + processor (for pre-processing the audio)
model = load_model("base", device=device).wav_encoder
model.eval()

processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

dataset = dataset.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
# dataset = dataset.with_format("torch", device=device)

# cast the audio data to the correct sampling rate for the model
for sample in tqdm(dataset):
    path = Path(sample["audio"]["path"])
    output_path = output_dir / path.parent.name / path.stem
    if output_path.exists():
        continue
    inputs = processor(
        raw_audio=sample["audio"]["array"],
        sampling_rate=processor.sampling_rate,
        return_tensors="pt",
    )

    features = (
        model(inputs["input_values"].squeeze(0).to(device))
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, features)
