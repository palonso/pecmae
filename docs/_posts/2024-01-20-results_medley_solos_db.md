---
layout: post
title: Medley-Solos-DB results
permalink: /medleysolosdb/
---



## Baselines

Our baselines consist on random samples from the training set of the Medley-solos-DB dataset and prototypes obtained with the [APNet](https://github.com/pzinemanas/APNet) model.

|---|---|---|
|class | dataset sample | APNet |
|---|---|---|
| clarinet | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/samples/Medley-solos-DB_training-0_0334342a-d60a-58d7-fdd3-336d304471ec.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/apnet/01_clarinet.wav?" controls preload></audio> |
| distorted electric guitar | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/samples/Medley-solos-DB_training-1_00352223-57bc-5a6e-fbee-4b17e4c499f6.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/apnet/03_distorted electric guitar.wav?" controls preload></audio> |
| female singer | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/samples/Medley-solos-DB_training-2_015a3c20-d642-56b2-f97a-a601a0bd2c69.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/apnet/02_female singer.wav?" controls preload></audio> |
| flute | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/samples/Medley-solos-DB_training-3_00fa5f4e-3114-58a4-f829-f13c30f59946.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/apnet/15_flute.wav?" controls preload></audio> |
| piano | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/samples/Medley-solos-DB_training-4_008bc279-90a2-5de5-fe5a-ab011cef41a1.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/apnet/16_piano.wav?" controls preload></audio> |
| tenor saxophone | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/samples/Medley-solos-DB_training-5_02dd9a9b-e5c0-532a-f058-f4ee59e0cf94.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/apnet/05_tenor saxophone.wav?" controls preload></audio> |
| trumpet | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/samples/Medley-solos-DB_training-6_00bc0b46-468d-54dd-fbd5-44eed8df2b04.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/apnet/06_trumpet.wav?" controls preload></audio> |
| violin | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/samples/Medley-solos-DB_training-7_0025e852-0a5c-54d5-fe8d-c9aabd72ff4a.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/apnet/34_violin.wav?" controls preload></audio> |
|---|---|---|

## Our models

We show the results obtained with PECMAE-3 (3 prototypes per target class).
For each class, we sonify two of the prototypes.

|---|---|---|
|class | PECMAE-3 (prototype 0) | PECMAE-3 (prototype 1) |
|---|---|---|
| clarinet | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_clarinet_n0_gs1.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_clarinet_n1_gs1.wav?" controls preload> |
| distorted electric guitar | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_distorted electric guitar_n0_gs1.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_distorted electric guitar_n1_gs1.wav?" controls preload> |
| female singer | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_female singer_n0_gs1.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_female singer_n1_gs1.wav?" controls preload> |
| flute | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_flute_n0_gs1.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_flute_n1_gs1.wav?" controls preload> |
| piano | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_piano_n0_gs1.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_piano_n1_gs1.wav?" controls preload> |
| tenor saxophone | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_tenor saxophone_n0_gs1.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_tenor saxophone_n1_gs1.wav?" controls preload> |
| trumpet | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_trumpet_n0_gs1.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_trumpet_n1_gs1.wav?" controls preload> |
| violin | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_violin_n0_gs1.wav?" controls preload></audio> | <audio src="https://raw.githubusercontent.com/palonso/pecmae-samples/main/medley_solos_db/pecmae-3/v489_violin_n1_gs1.wav?" controls preload> |
|---|---|---|

### Prototype-class connections
In PECMAE, prototypes are linearly connected to the classification layer.
The following plot shows the weights learned for these connections.
Note that certain prototypes may have a positive correlation with related classes.
For example, `flute` and `clarinet`.

![](/pecmae/assets/images/Medley-Solos-DB_lin_weights.png)
