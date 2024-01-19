from pathlib import Path
from argparse import ArgumentParser

from datasets import load_dataset, Audio
from transformers import AutoProcessor
from tqdm import tqdm
import numpy as np

from encodecmae import load_model


parser = ArgumentParser()
parser.add_argument("audio_dir", type=Path)
parser.add_argument("embeddings_dir", type=Path)
parser.add_argument("--model-size", type=str, default="base")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--force", action="store_true")
parser.add_argument("--aggregate-time", type=int, default=-1)

args = parser.parse_args()

force = args.force
aggregate_time = args.aggregate_time

dataset = load_dataset("audiofolder", data_dir=args.audio_dir)

mae = load_model(args.model_size, device=args.device)
mae.eval()


for split_name in dataset.keys():
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    split = dataset[split_name].cast_column(
        "audio", Audio(sampling_rate=processor.sampling_rate)
    )

    # cast the audio data to the correct sampling rate for the model
    for sample in tqdm(split):
        path = Path(sample["audio"]["path"])
        output_path = (
            args.embeddings_dir / args.model_size / path.parent.name / path.stem
        )

        if output_path.exists() and not force:
            continue

        inputs = processor(
            raw_audio=sample["audio"]["array"],
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        )

        features = mae.extract_features_from_array(
            inputs["input_values"].squeeze().to(args.device)
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        features = features.squeeze()
        if aggregate_time > 0:
            last_time = features.shape[0] % aggregate_time
            features = features[:-last_time, :].reshape(
                -1, aggregate_time, features.shape[-1]
            )
            features = features.mean(axis=1).squeeze()

        np.save(output_path, features)
