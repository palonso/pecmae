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

#### Baselines

|---|---|---|
|class | dataset sample | APNet |
|---|---|---|
| clarinet | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/samples/Medley-solos-DB_training-0_0334342a-d60a-58d7-fdd3-336d304471ec.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/apnet/01_clarinet.wav?raw=true" controls preload></audio> |
| distorted electric guitar | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/samples/Medley-solos-DB_training-1_00352223-57bc-5a6e-fbee-4b17e4c499f6.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/apnet/03_distorted electric guitar.wav?raw=true" controls preload></audio> |
| female singer | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/samples/Medley-solos-DB_training-2_015a3c20-d642-56b2-f97a-a601a0bd2c69.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/apnet/02_female singer.wav?raw=true" controls preload></audio> |
| flute | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/samples/Medley-solos-DB_training-3_00fa5f4e-3114-58a4-f829-f13c30f59946.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/apnet/15_flute.wav?raw=true" controls preload></audio> |
| piano | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/samples/Medley-solos-DB_training-4_008bc279-90a2-5de5-fe5a-ab011cef41a1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/apnet/16_piano.wav?raw=true" controls preload></audio> |
| tenor saxophone | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/samples/Medley-solos-DB_training-5_02dd9a9b-e5c0-532a-f058-f4ee59e0cf94.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/apnet/05_tenor saxophone.wav?raw=true" controls preload></audio> |
| trumpet | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/samples/Medley-solos-DB_training-6_00bc0b46-468d-54dd-fbd5-44eed8df2b04.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/apnet/06_trumpet.wav?raw=true" controls preload></audio> |
| violin | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/samples/Medley-solos-DB_training-7_0025e852-0a5c-54d5-fe8d-c9aabd72ff4a.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/apnet/34_violin.wav?raw=true" controls preload></audio> |
|---|---|---|

#### Our models

|---|---|---|
|class | PECMAE-3 (prototype 0) | PECMAE-3 (prototype 1) |
|---|---|---|
| clarinet | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_clarinet_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_clarinet_n1_gs1.wav?raw=true" controls preload> |
| distorted electric guitar | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_distorted electric guitar_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_distorted electric guitar_n1_gs1.wav?raw=true" controls preload> |
| female singer | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_female singer_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_female singer_n1_gs1.wav?raw=true" controls preload> |
| flute | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_flute_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_flute_n1_gs1.wav?raw=true" controls preload> |
| piano | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_piano_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_piano_n1_gs1.wav?raw=true" controls preload> |
| tenor saxophone | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_tenor saxophone_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_tenor saxophone_n1_gs1.wav?raw=true" controls preload> |
| trumpet | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_trumpet_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_trumpet_n1_gs1.wav?raw=true" controls preload> |
| violin | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_violin_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/medley_solos_db/pecmae-3/v489_violin_n1_gs1.wav?raw=true" controls preload> |
|---|---|---|
