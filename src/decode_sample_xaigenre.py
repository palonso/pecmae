import functools
from argparse import ArgumentParser
from pathlib import Path
from functools import partial
from math import pi

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from encodec import EncodecModel
from essentia.standard import MonoWriter
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from encodecmae import load_model
from encodecmae.tasks.models.transformers import (
    TransformerEncoder,
    MultiHeadAttention,
    SinusoidalPositionalEmbeddings,
)


from labelmaps import xaigenre_label2id

sr = 24000


model_map = {
    "base": 768,
    "large": 1024,
}

parser = ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--output_dir", type=Path, default="out_data/decoded/")
args = parser.parse_args()

model_size = "large_diffusion"
device = args.device
output_dir = args.output_dir

file = "selected-genres.spotifyapi.clean.tsv.train"
base = Path("feats_xai_genre_v1_diff/large_diffusion/audio/")

df = pd.read_csv(file, sep="\t", header=None)
genres = list(xaigenre_label2id.keys())

if model_size == "base":

    class EnCodecMAEtoEnCodec(pl.LightningModule):
        def __init__(
            self, model_dim=768, encodec_dim=128, num_decoder_layers=4, num_heads=12
        ):
            super().__init__()
            self.decoder = TransformerEncoder(
                model_dim=model_dim,
                num_layers=num_decoder_layers,
                attention_layer=functools.partial(
                    MultiHeadAttention, model_dim=model_dim, num_heads=num_heads
                ),
                compile=False,
            )
            self.decoder_proj = torch.nn.Linear(model_dim, encodec_dim)

        def forward(self, x):
            dec_out = self.decoder(x, padding_mask=None)
            y = self.decoder_proj(dec_out)
            return y

    ecmae = load_model(model_size, device=device)
    ecmae2ec = EnCodecMAEtoEnCodec(model_dim=model_map[model_size])
    ckpt_file = hf_hub_download(
        repo_id=f"lpepino/encodecmae-{model_size}", filename="ecmae2ec.pt"
    )
    ckpt = torch.load(ckpt_file, map_location=device)
    ecmae2ec.load_state_dict(ckpt, strict=False)
    ecmae2ec.to(device)
    ecmae2ec.eval()

elif model_size == "large":

    class EnCodecMAEtoEnCodec(pl.LightningModule):
        def __init__(
            self, model_dim=768, encodec_dim=128, num_decoder_layers=1, num_heads=12
        ):
            super().__init__()
            self.posenc = SinusoidalPositionalEmbeddings(model_dim)
            self.trans = TransformerEncoder(
                model_dim=model_dim,
                num_layers=num_decoder_layers,
                attention_layer=functools.partial(
                    MultiHeadAttention, model_dim=model_dim, num_heads=num_heads
                ),
                compile=False,
            )
            self.lin = torch.nn.Linear(model_dim, encodec_dim)

        def forward(self, x, padding_mask=None):
            y = self.posenc(x)
            y = self.trans(y, padding_mask=padding_mask)
            return self.lin(y)

    ecmae = load_model(model_size, device=device)
    ecmae2ec = EnCodecMAEtoEnCodec(model_dim=model_map[model_size])
    ckpt_file = hf_hub_download(
        repo_id=f"lpepino/encodecmae-{model_size}", filename="ecmae2ec.pt"
    )
    ckpt = torch.load(ckpt_file, map_location=device)
    ecmae2ec.load_state_dict(ckpt, strict=False)
    ecmae2ec.to(device)
    ecmae2ec.eval()

elif model_size in ("base_diffusion", "large_diffusion"):

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
        diffusion_steps = 35
        guidance_strength = 1
        win_size = 24000
        hop_size = 24000
        model_filename = (
            "ae/diffusion_encodec_transformer_4L8Lclsx1_punc01_250ksteps.ckpt"
        )

    elif model_size == "large_diffusion":
        diffusion_steps = 40
        guidance_strength = 1
        win_size = 24000 * 4
        hop_size = 24000 * 4
        model_filename = (
            "ae/diffusion_encodec_transformer_4L8Lclsx1_punc01_330ksteps.ckpt"
        )

    ckpt_file = hf_hub_download(
        repo_id="lpepino/encodecmae-base",
        filename=model_filename,
    )

    ecmae = TransformerAEDiffusion(
        num_cls_tokens=1,
        signal_to_generate="wav_features",
        signal_dim=128,
        num_denoiser_layers=8,
    )
    ecmae.load_state_dict(torch.load(ckpt_file, map_location=device)["state_dict"])
    ecmae.to(device)


else:
    raise ValueError(f"Unknown model size: {model_size}")


ec = EncodecModel.encodec_model_24khz()
ec.to(device)
for genre in genres:
    df_genre = df[df[0] == genre]
    sid = df_genre.iloc[0, 1]
    base = Path("/mnt/projects/xai-workshop-genre-dataset/v1/spotify-audio/")
    filename = base / genre / "audio" / f"{sid}.mp3"
    import shutil

    shutil.copy(filename, output_dir / f"{genre}_orig.mp3")
    continue
    data = np.load(filename)

    encmae_feats = torch.tensor(data, device=device)

    if model_size in ("base_diffusion", "large_diffusion"):
        audio = np.zeros((encmae_feats.shape[0] - 1) * hop_size + win_size)

        for i in tqdm(range(encmae_feats.shape[0])):
            with torch.no_grad():
                rec = ecmae.sample(
                    encmae_feats[i].unsqueeze(0),
                    steps=diffusion_steps,
                    guidance_strength=guidance_strength,
                    length=win_size * 75 // 24000,
                )
                ecmae_features = rec[-1]
                reconstruction = ec.decoder(ecmae_features.transpose(1, 2))
                audio[i * hop_size : i * hop_size + win_size] += (
                    audio[i * hop_size : i * hop_size + win_size]
                    + reconstruction[0, 0].detach().cpu().numpy()
                )

    elif model_size in ("base", "large"):
        with torch.no_grad():
            ec_dec_in = ecmae2ec(encmae_feats)
            reconstruction = ec.decoder(ec_dec_in.transpose(1, 2))

        audio = reconstruction.cpu().numpy().squeeze()

        if len(audio.shape) > 1:
            audio = audio.flatten()

    out_path = output_dir / f"{genre}_sample.wav"

    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True)

    MonoWriter(filename=str(out_path), sampleRate=sr)(audio)
