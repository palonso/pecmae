# Prototype EnCodecMAE (PECMAE)

Code for the paper [Leveraging Pre-trained Autoencoders for Interpretable Prototype Learning of Music Audio](https://arxiv.org/abs/2402.09318).

The autoencoder code is available in [this repository](https://github.com/mrpep/encodecmae-to-wav).

Sonification results available in the [companion web site](https://palonso.github.io/pecmae/).



## Setup

1. Clone the repository

```bash
git clone git@github.com:palonso/pecmae.git
```

2. Initialize the submodules

```bash
git submodule update --init --recursive
```

3. Create a virtual environment (recommended)

```bash
python -m venv venv && source venv/bin/activate
```

4. Install the requirements

```bash
TODO
```

---
**NOTE**
The code is tested with Python 3.11

---


## Experiments

### Pre-processing

1. Download GTZAN

```bash
python src/download.py --dataset gtzan 
```


### Train PECMAE

1. Extract AE EnCodecMAE features

```bash
python src/encode_encodecmae.py \
  /Users/palonso/reps/pecmae/data/downloads/extracted/6e1d1c4caea7b374d35053b71d94153382362d70b0c1ca1e39628dd945f0ded7/genres \
  feats/gtzan/ \
  --model-size large_difussion
```

3. Train Prototypical Network

## Adding a new Dataset

## Citation
If results, insights, or code developed within this project are useful for you, please consider citing our work:

    @inproceedings{alonso2024leveraging,
      author    = "Alonso-Jim\'{e}nez, Pablo and Pepino, Leonardo and Batlle-Roca, Roser and Zinemanas, Pablo and Bogdanov, Dmitry and Serra, Xavier and Rocamora, Mart\'{i}n",
      title     = "Leveraging Pre-trained Autoencoders for Interpretable Prototype Learning of Music Audio",
      maintitle = "IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)",
      booktitle = "ICASSP Workshop on Explainable AI for Speech and Audio (XAI-SA)",
      year      = 2024,
    }
