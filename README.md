# Prototype EnCodecMAE (PECMAE)

Code for the paper [Leveraging Pre-trained Autoencoders for Interpretable Prototype Learning of Music Audio](https://arxiv.org/abs/2402.09318).

The autoencoder code is available in [this repository](https://github.com/mrpep/encodecmae-to-wav).

Sonification results available in the [companion web site](https://palonso.github.io/pecmae/).



## Setup

1. Create a virtual environment (recommended):

```bash
python -m venv venv && source venv/bin/activate

```

2. Initialize submodules and install dependencies:

```bash
./setup.sh

```

---
**NOTE**
The setup scrupt was only tested with Python 3.11 using CentOS 7.5
---


## Experiments

### Pre-processing

1. Download a dataset (e.g., GTZAN):

```bash
python src/download.py --dataset gtzan 

```

2. Extract the conditioning features:

```bash
python src/encode_encodecmae.py audio/gtzan/ feats/gtzan/ --model diffusion_4s
```

The available options are: `base`, `large`, `diffusion_1s`, `diffusion_4s`, and `diffusion_10s`.
`base`, and `large` are EnCodecMAE embeddings (not intended to operate with the diffusion decoder).
`diffusion_4s` is the model that we used in the paper, and `diffusion_10s` is a newer version that was not included in the website, but we provide sonification examples in the companion website.


### Train PECMAE

1. Train Prototypical Network

```bash
./scripts/train_pecmae_5_gtzan.sh

```

2. Train the baseline models

```bash
TODO

```

## Using PECMAE with your data

To use PECMAE with your custom dataset, follow these steps

1. Given an audio dataset located at `/your/dataaset/`, extract the conditioning features

```bash
python src/encode_encodecmae.py /your/dataset/ feats/your_dataset/ --model diffusion_4s --audio-format .your_dataset_audio_extension

```

2. Create a training script similar to `./scripts/train_pecmae_5_gtzan.sh`.

Your should modify the fields `--data-dir`, `--metadata-file-train`, `--metadata-file-val`, and `--metadata-file-test` to point to your groundtruth file.
Have a look at `./groundtruth/` to see examples of the expected format.


## Sonifying your prototypes

TODO


## Citation
If results, insights, or code developed within this project are useful for you, please consider citing our work:

    @inproceedings{alonso2024leveraging,
      author    = "Alonso-Jim\'{e}nez, Pablo and Pepino, Leonardo and Batlle-Roca, Roser and Zinemanas, Pablo and Bogdanov, Dmitry and Serra, Xavier and Rocamora, Mart\'{i}n",
      title     = "Leveraging Pre-trained Autoencoders for Interpretable Prototype Learning of Music Audio",
      maintitle = "IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)",
      booktitle = "ICASSP Workshop on Explainable AI for Speech and Audio (XAI-SA)",
      year      = 2024,
    }
