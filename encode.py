from pathlib import Path

from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor
from tqdm import tqdm
import numpy as np


audio_dir = Path("/mnt/mtgdb-audio/stable/genre_tzanetakis/audio/22kmono/")
dataset = load_dataset("audiofolder", data_dir=audio_dir, split="train")


# dummy dataset, however you can swap this with a dataset on the 🤗 hub or bring your own

# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

dataset = dataset.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

# cast the audio data to the correct sampling rate for the model
for sample in tqdm(dataset):
    path = Path(sample["audio"]["path"])
    output_path = Path("codes") / path.parent.name / path.stem
    if output_path.exists():
        continue
    inputs = processor(
        raw_audio=sample["audio"]["array"],
        sampling_rate=processor.sampling_rate,
        return_tensors="pt",
    )
    codes = model.encode(inputs["input_values"], inputs["padding_mask"]).audio_codes
    codes = codes.detach().numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, codes)
