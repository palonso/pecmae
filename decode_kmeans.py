import functools
from argparse import ArgumentParser
from pathlib import Path
import pickle as pk

import torch
import numpy as np
import pytorch_lightning as pl
from encodec import EncodecModel
from essentia.standard import MonoWriter
from huggingface_hub import hf_hub_download

from encodecmae import load_model
from encodecmae.tasks.models.transformers import (
    TransformerEncoder,
    MultiHeadAttention,
    SinusoidalPositionalEmbeddings,
)

sr = 24000


label_map = {
    0: "blu",
    1: "cla",
    2: "cou",
    3: "dis",
    4: "hip",
    5: "jaz",
    6: "met",
    7: "pop",
    8: "reg",
    9: "roc",
}

label_map = {
    0: "bass",
    1: "brass",
    2: "flute",
    3: "guitar",
    4: "keyboard",
    5: "mallet",
    6: "organ",
    7: "reed ",
    8: "string",
    9: "volcal",
}

model_map = {
    "base": 768,
    "large": 1024,
}

parser = ArgumentParser()
parser.add_argument("k_means_file", type=str)
parser.add_argument("--model-size", type=str, default="base")
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--output_dir", type=Path, default="out_data/decoded/")
parser.add_argument(
    "--data-type", type=str, choices=["centers", "samples"], default="centers"
)
args = parser.parse_args()

k_means_file = args.k_means_file
model_size = args.model_size
device = args.device
output_dir = args.output_dir
data_type = args.data_type


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

else:
    raise ValueError(f"Unknown model size: {model_size}")

ecmae = load_model(model_size, device=device)
ecmae2ec = EnCodecMAEtoEnCodec(model_dim=model_map[model_size])
ckpt_file = hf_hub_download(
    repo_id=f"lpepino/encodecmae-{model_size}", filename="ecmae2ec.pt"
)
ckpt = torch.load(ckpt_file, map_location=device)
ecmae2ec.load_state_dict(ckpt, strict=False)
ec = EncodecModel.encodec_model_24khz()
ecmae2ec.to(device)
ec.to(device)
ecmae2ec.eval()

with open(k_means_file, "rb") as f:
    data = pk.load(f)


protos_per_label = 1
label_i = 0
for i in range(len(label_map)):
    print(f"class {label_map[label_i]}")
    encmae_feats = torch.tensor(
        data[f"label.{i}.{data_type}"].reshape(-1, 768), device=device
    ).unsqueeze(0)

    with torch.no_grad():
        ec_dec_in = ecmae2ec(encmae_feats)
        reconstruction = ec.decoder(ec_dec_in.transpose(1, 2))

    audio = reconstruction.cpu().numpy().squeeze()

    out_path = (
        output_dir
        / f"k_means_{data_type}_{label_map[label_i]}_n{(i+ 1) % protos_per_label}.wav"
    )

    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True)

    MonoWriter(filename=str(out_path), sampleRate=sr)(audio)

    if (i + 1) % protos_per_label == 0:
        label_i += 1
