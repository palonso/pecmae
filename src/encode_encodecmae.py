from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from math import pi
from glob import glob

import librosa
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import pytorch_lightning as pl

from encodecmae import load_model
from encodecmae.tasks.models.transformers import (
    TransformerEncoder,
    MultiHeadAttention,
    SinusoidalPositionalEmbeddings,
)


parser = ArgumentParser()
parser.add_argument("audio_dir", type=Path)
parser.add_argument("embeddings_dir", type=Path)
parser.add_argument(
    "--model-size",
    type=str,
    default="base",
    choices=[
        "base",
        "base_diffusion",
        "large",
        "large_diffusion",
        "large_diffusion_10s",
    ],
)
parser.add_argument("--format", type=str, default="wav", choices=["wav", "mp3"])
parser.add_argument("--device", type=str, default="cuda:0")

args = parser.parse_args()

device = args.device
model_size = args.model_size

if model_size in ("base_diffusion", "large_diffusion", "large_diffusion_10s"):

    class TransformerCLSEncoder(torch.nn.Module):
        def __init__(
            self,
            encodecmae_model="base",
            num_heads=12,
            num_encoder_layers=2,
            num_cls_tokens=1,
            device="cpu",
            downsample_factor=75,
        ):
            super().__init__()
            self.encodecmae_model = load_model(encodecmae_model, device=device)
            self.encodecmae_model.visible_encoder.compile = False
            model_dim = self.encodecmae_model.visible_encoder.model_dim
            self.encoder = TransformerEncoder(
                model_dim,
                attention_layer=partial(
                    MultiHeadAttention, model_dim=model_dim, num_heads=num_heads
                ),
                num_layers=2,
                compile=False,
            )
            self.cls_tokens = torch.nn.Embedding(num_cls_tokens, model_dim)
            self.pos_encoder = SinusoidalPositionalEmbeddings(model_dim)
            self.num_cls_tokens = num_cls_tokens
            self.out_channels = model_dim
            self.downsample_factor = downsample_factor

        def forward(self, x):
            with torch.no_grad():
                self.encodecmae_model.encode_wav(x)
                self.encodecmae_model.mask(x, ignore_mask=True)
                self.encodecmae_model.encode_visible(x)
            cls_tokens = torch.tile(
                self.cls_tokens(
                    torch.arange(
                        self.num_cls_tokens, device=x["visible_embeddings"].device
                    ).unsqueeze(0)
                ),
                (x["visible_embeddings"].shape[0], 1, 1),
            )
            enc_in = torch.cat(
                [cls_tokens, self.pos_encoder(x["visible_embeddings"])], dim=1
            )
            padding_mask = torch.cat(
                [
                    torch.zeros(
                        (x["visible_embeddings"].shape[0], self.num_cls_tokens),
                        device=x["visible_embeddings"].device,
                    ),
                    x["feature_padding_mask"],
                ],
                dim=1,
            )
            enc_out = self.encoder(enc_in, padding_mask)
            cls = enc_out[:, : self.num_cls_tokens]
            x["cls_token"] = cls
            return x

    class TransformerAEDiffusion(pl.LightningModule):
        def __init__(
            self,
            encodecmae_model="base",
            num_cls_tokens=5,
            num_encoder_layers=4,
            num_denoiser_layers=4,
            num_heads=8,
            signal_to_generate="visible_embeddings",
            signal_dim=768,
        ):
            super().__init__()
            self.encoder = TransformerCLSEncoder(
                num_cls_tokens=num_cls_tokens,
                num_encoder_layers=num_encoder_layers,
                encodecmae_model=encodecmae_model,
                num_heads=num_heads,
            )
            self.denoiser = TransformerEncoder(
                self.encoder.out_channels,
                attention_layer=partial(
                    MultiHeadAttention,
                    model_dim=self.encoder.out_channels,
                    num_heads=num_heads,
                ),
                num_layers=num_denoiser_layers,
                compile=False,
            )
            self.in_adapter = torch.nn.Linear(signal_dim, self.encoder.out_channels)
            self.out_adapter = torch.nn.Linear(self.encoder.out_channels, signal_dim)
            self.time_embedding = torch.nn.Sequential(
                torch.nn.Linear(1, self.encoder.out_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(self.encoder.out_channels, self.encoder.out_channels),
            )
            self.signal_to_generate = signal_to_generate
            self.signal_dim = signal_dim

        def encode(self, x):
            return self.encoder(x)

        def sample(self, code, steps=10, length=75, guidance_strength=0):
            with torch.no_grad():
                sigmas = torch.linspace(1, 0, steps + 1, device=code.device)
                ts = self.time_embedding(sigmas[:, None, None])
                angle = sigmas * pi / 2
                alphas, betas = torch.cos(angle), torch.sin(angle)
                x_noisy = torch.randn((1, length, self.signal_dim), device=code.device)
                noise_sequence = []
                for i in range(steps):
                    denoiser_in = torch.cat(
                        [
                            code,
                            ts[i, :, :].unsqueeze(0),
                            self.encoder.pos_encoder(self.in_adapter(x_noisy)),
                        ],
                        dim=1,
                    )
                    v_pred = self.out_adapter(
                        self.denoiser(denoiser_in, None)[
                            :, self.encoder.num_cls_tokens + 1 :
                        ]
                    )
                    if guidance_strength > 0:
                        denoiser_in_unc = torch.cat(
                            [
                                torch.zeros_like(code),
                                ts[i, :, :].unsqueeze(0),
                                self.encoder.pos_encoder(self.in_adapter(x_noisy)),
                            ],
                            dim=1,
                        )
                        v_pred_unc = self.out_adapter(
                            self.denoiser(denoiser_in_unc, None)[
                                :, self.encoder.num_cls_tokens + 1 :
                            ]
                        )
                        v_pred += guidance_strength * (v_pred - v_pred_unc)
                    x_pred = alphas[i] * x_noisy - betas[i] * v_pred
                    noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
                    x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
                    noise_sequence.append(x_noisy)
            return noise_sequence

    if model_size == "base_diffusion":
        model_filename = (
            "ae/diffusion_encodec_transformer_4L8Lclsx1_punc01_250ksteps.ckpt"
        )
        num_encoder_layers = 4
        diffusion_steps = 35
        guidance_strength = 1
        win_size = 24000
        hop_size = 24000

    elif model_size == "large_diffusion":
        model_filename = (
            "ae/diffusion_encodec_transformer_4L8Lclsx1_punc01_330ksteps.ckpt"
        )
        num_encoder_layers = 4
        diffusion_steps = 40
        guidance_strength = 1
        win_size = 24000 * 4
        hop_size = 24000 * 4

    elif model_size == "large_diffusion_10s":
        model_filename = "ae/diffusion-10s-2L8L-jamendofma-285k.ckpt"
        num_encoder_layers = 2
        diffusion_steps = 40
        guidance_strength = 1
        win_size = 24000 * 10
        hop_size = 24000 * 10

    ckpt_file = hf_hub_download(
        repo_id="lpepino/encodecmae-base",
        filename=model_filename,
    )

    ecmae = TransformerAEDiffusion(
        num_cls_tokens=1,
        signal_to_generate="wav_features",
        signal_dim=128,
        num_encoder_layers=num_encoder_layers,
        num_denoiser_layers=8,
    )
    ecmae.load_state_dict(torch.load(ckpt_file, map_location=device)["state_dict"])
    ecmae.to(device)


else:
    ecmae = load_model("base", device=device)

# ec = EncodecModel.encodec_model_24khz()
# ec.to(device)

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

    if model_size in ("base_diffusion", "large_diffusion", "large_diffusion_10s"):
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
