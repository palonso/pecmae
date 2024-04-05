---
layout: post
title: Prototype EnCodecMAE (PECMAE)
---

This site contains sonification of the prototypes obtained with PECMAE models.
Aditionally, we provide baselines consisting of a random sample from the training set of each dataset and sonifacions obtained with an [APNet](https://github.com/pzinemanas/APNet/) model.


## Updates

- ⚠️ **2024/02/02. Fixed  prototype URLs in GTZAN**

    > By mistake we duplicated the links for prototypes 0 and 1, and they sounded identical. Fixed now

- **2024/01/22. Newer PECMAE model with 10-second context**

    > The sonification results obtained with model PECMAE-5 (10s) are based on a newer autoencoder featuring the same architecture presented in the paper but using a longer context window of 10 seconds.
Results suggest that extending the receptive field allows to achieve prototypes that better resemble their classes.


## Citation

If results, insights, or code developed within this project are useful for you, please consider citing our work:

    @inproceedings{alonso2024leveraging,
      author    = "Alonso-Jim\'{e}nez, Pablo and Pepino, Leonardo and Batlle-Roca, Roser and Zinemanas, Pablo and Bogdanov, Dmitry and Serra, Xavier and Rocamora, Mart\'{i}n",
      title     = "Leveraging Pre-trained Autoencoders for Interpretable Prototype Learning of Music Audio",
      maintitle = "IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)",
      booktitle = "ICASSP Workshop on Explainable AI for Speech and Audio (XAI-SA)",
      year      = 2024,
    }


## Acknowledgments

This work has been supported by the [Musical AI project](https://www.upf.edu/web/mtg/ongoing-projects/-/asset_publisher/DneGVrJZ7tmE/content/id/235850570/) - ``PID2019-111403GB-I00/AEI/10.13039/501100011033``, funded by the [Spanish Ministerio de Ciencia e Innovación](https://www.ciencia.gob.es/en/) and the [Agencia Estatal de Investigación](https://www.aei.gob.es/).
