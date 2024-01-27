---
layout: default
title: Home
---


## Sonification of the PECMAE prototypes

This site contains sonification of the prototypes obtained with PECMAE models.
Aditionally, we provide baselines samples consisting of a random sample from the training set and sonifacions obtained with an [APNet](https://github.com/pzinemanas/APNet/) model.


``Note: the model PECMAE-5 (10s) is based on a newer autoencoder featuring the same architecture presented in the paper but using a longer context window of 10 seconds. Results suggest that extending the receptive field allows to achieve prototypes that better resemble their classes.``

### GTZAN

#### Baselines


|---|---|---|
|class | dataset sample | APNet |
|---|---|---|
| blues | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/samples/blues.00000.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/apnet/00_blues.wav?raw=true" controls preload></audio> |
| classical | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/samples/classical.00000.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/apnet/01_classical.wav?raw=true" controls preload></audio> |
| country | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/samples/country.00000.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/apnet/02_country.wav?raw=true" controls preload></audio> |
| disco | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/samples/disco.00000.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/apnet/03_disco.wav?raw=true" controls preload></audio> |
| hip-hop | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/samples/hip-hop.00000.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/apnet/04_hip-hop.wav?raw=true" controls preload></audio> |
| jazz | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/samples/jazz.00000.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/apnet/05_jazz.wav?raw=true" controls preload></audio> |
| metal | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/samples/metal.00000.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/apnet/06_metal.wav?raw=true" controls preload></audio> |
| pop | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/samples/pop.00000.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/apnet/07_pop.wav?raw=true" controls preload></audio> |
| reggae | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/samples/reggae.00000.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/apnet/08_reggae.wav?raw=true" controls preload></audio> |
| rock | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/samples/rock.00000.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/apnet/09_rock.wav?raw=true" controls preload></audio> |
|---|---|---|

#### Our models

|---|---|---|
|class | PECMAE-3 | PECMAE-5 (10s) |
|---|---|---|
| blues | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-3/v491_blu_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-5-10s/v533_blu_n0_gs1.wav?raw=true" controls preload></audio> |
| classical | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-3/v491_cla_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-5-10s/v533_cla_n0_gs1.wav?raw=true" controls preload></audio> |
| country | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-3/v491_cou_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-5-10s/v533_cou_n0_gs1.wav?raw=true" controls preload></audio> |
| disco | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-3/v491_dis_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-5-10s/v533_dis_n0_gs1.wav?raw=true" controls preload></audio> |
| hip-hop | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-3/v491_hip_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-5-10s/v533_hip_n0_gs1.wav?raw=true" controls preload></audio> |
| jazz | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-3/v491_jaz_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-5-10s/v533_jaz_n0_gs1.wav?raw=true" controls preload></audio> |
| metal | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-3/v491_met_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-5-10s/v533_met_n0_gs1.wav?raw=true" controls preload></audio> |
| pop | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-3/v491_pop_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-5-10s/v533_pop_n0_gs1.wav?raw=true" controls preload></audio> |
| reggae | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-3/v491_reg_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-5-10s/v533_reg_n0_gs1.wav?raw=true" controls preload></audio> |
| rock | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-3/v491_roc_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/gtzan/pecmae-5-10s/v533_roc_n0_gs1.wav?raw=true" controls preload></audio> |
|---|---|---|

### XAI-Genre

TODO

#### Baselines

#### Our models

### Medley-Solos-DB

TODO

#### Baselines

#### Our models
