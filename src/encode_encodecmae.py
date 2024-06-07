from argparse import ArgumentParser
from pathlib import Path
from glob import glob

import librosa
import numpy as np
import torch
from tqdm import tqdm

from encodecmae_to_wav.hub import load_model
import encodecmae


parser = ArgumentParser()
parser.add_argument("audio_dir", type=Path)
parser.add_argument("embeddings_dir", type=Path)
parser.add_argument(
    "--model",
    type=str,
    default="base",
    choices=[
        "base",
        "large",
        "base_diffusion",
        "large_diffusion",
        "large_diffusion_10s",
    ],
)
parser.add_argument("--format", type=str, default="wav", choices=["wav", "mp3"])
parser.add_argument("--device", type=str, default="cuda:0")

args = parser.parse_args()

device = args.device
model = args.model

if model in ("base_diffusion", "large_diffusion", "large_diffusion_10s"):
    if model == "base_diffusion":
        model_filename = "ecmae2ec-base-1LTransformer"

    elif model == "large_diffusion":
        model_filename = "DiffTransformerAE2L8L1CLS-4s"

    elif model == "large_diffusion_10s":
        model_filename = "DiffTransformerAE2L8L1CLS-10s"

    ecmae = load_model(model_filename, device=device)

else:
    ecmae = encodecmae.load_model(model, device=device)

glob_pattern = str(args.audio_dir / "**" / f"*.{args.format}")
print(glob_pattern)
audio_files = glob(glob_pattern, recursive=True)

for audio_file in tqdm(audio_files):
    audio_file = Path(audio_file)

    output_path = (
        args.embeddings_dir / args.model_size / audio_file.parent.name / audio_file.name
    ).with_suffix(".npy")

    if output_path.exists():
        continue

    xorig, fs = librosa.core.load(audio_file, sr=24000)

    if model in ("base_diffusion", "large_diffusion", "large_diffusion_10s"):
        ecmae_feature_stack = []
        for i in range(0, max(len(xorig) - win_size, 1), hop_size):
            x = {
                "wav": torch.from_numpy(xorig[i : i + win_size])
                .unsqueeze(0)
                .to(device),
                "wav_lens": torch.tensor(
                    [
                        win_size,
                    ],
                    device=device,
                ),
            }
            with torch.no_grad():
                ecmae.encode(x)
                ecmae_feature_stack.append(x["cls_token"])
        ecmae_features = torch.cat(ecmae_feature_stack, dim=0)

    else:
        ecmae.visible_encoder.compile = False

        n_patches = xorig.shape[0] // (4 * fs)
        xorig = xorig[: n_patches * 4 * fs].reshape(-1, 4 * fs)

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
            ecmae.encode_wav(x)
            ecmae.mask(x, ignore_mask=True)
            ecmae.encode_visible(x)
            ecmae_features = x["visible_embeddings"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, ecmae_features.detach().cpu().numpy())
