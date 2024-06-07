from argparse import ArgumentParser
from pathlib import Path
from glob import glob

import librosa
import numpy as np
import torch
from tqdm import tqdm

from encodecmae_to_wav.hub import load_model
import encodecmae


sr = 24000

def encode_encodecmae(
    audio_dir: Path,
    features_dir: Path,
    model: str,
    device="cuda:0",
    audio_format=".wav",
):

    # load the model
    if model in ("base", "large"):
        encoder = encodecmae.load_model(model, device=device)
    else:

        if model == "diffusion_1s":
            model_filename = "ecmae2ec-base-1LTransformer"
        elif model == "diffusion_4s":
            model_filename = "DiffTransformerAE2L8L1CLS-4s"
        elif model == "diffusion_10s":
            model_filename = "DiffTransformerAE2L8L1CLS-10s"
        else:
            raise ValueError(f"{model} is not implemented")

        encoder = load_model(model_filename, device=device)

    # find audio files
    glob_pattern = str(audio_dir / "**" / f"*{audio_format}")
    print("glob pattern:", glob_pattern)
    audio_files = glob(glob_pattern, recursive=True)

    for audio_file in tqdm(audio_files):
        try:
            audio_file = Path(audio_file)

            output_path = (
                features_dir / model / audio_file.parent.name / audio_file.name
            ).with_suffix(".npy")

            if output_path.exists():
                continue

            xorig, _ = librosa.core.load(audio_file, sr=sr)

            if model in ("base", "large"):
                encoder.visible_encoder.compile = False
                n_seconds = 4

                n_patches = xorig.shape[0] // (n_seconds * fs)
                xorig = xorig[: n_patches * n_seconds * fs].reshape(-1, n_seconds * fs)

                x = {
                    "wav": torch.from_numpy(xorig).to(device=device, dtype=ecmae.dtype),
                    "wav_lens": torch.tensor(
                        [
                            xorig.shape[1],
                        ]
                        * xorig.shape[0],
                        device=device,
                    ),
                }

                with torch.no_grad():
                    encoder.encode_wav(x)
                    encoder.mask(x, ignore_mask=True)
                    encoder.encode_visible(x)
                    features = x["visible_embeddings"]

            else:
                features = encoder.encode(xorig)


            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, features.detach().cpu().numpy())

        except Exception:
            print(f"`{audio_file}` failed")
            

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("features_dir", type=Path)
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=[
            "base",
            "large",
            "diffusion_1s",
            "diffusion_4s",
            "diffusion_10s",
        ],
    )
    parser.add_argument("--format", type=str, default=".wav", choices=[".wav", ".mp3"])
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    model = args.model

    encode_encodecmae(
        args.audio_dir,
        args.features_dir,
        model,
        device=args.device,
        audio_format=args.format,
    )
