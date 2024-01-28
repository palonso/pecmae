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
### XAI-Genre

#### Baselines

|---|---|---|
|class | dataset sample | APNet |
|---|---|---|
| afrobeat | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/afrobeat.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/00_afrobeat.wav?raw=true" controls preload></audio> |
| ambient | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/ambient.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/01_ambient.wav?raw=true" controls preload></audio> |
| bolero | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/bolero.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/04_bolero.wav?raw=true" controls preload></audio> |
| bop | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/bop.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/05_bop.wav?raw=true" controls preload></audio> |
| bossa nova | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/bossa_nova.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/08_bossa nova.wav?raw=true" controls preload></audio> |
| contemporary jazz | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/contemporary_jazz.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/00_contemporary jazz.wav?raw=true" controls preload></audio> |
| cumbia | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/cumbia.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/12_cumbia.wav?raw=true" controls preload></audio> |
| disco | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/disco.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/17_disco.wav?raw=true" controls preload></audio> |
| doo wop | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/doo_wop.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/19_doo wop.wav?raw=true" controls preload></audio> |
| dub | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/dub.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/27_dub.wav?raw=true" controls preload></audio> |
| electro | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/electro.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/33_electro.wav?raw=true" controls preload></audio> |
| europop | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/europop.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/34_europop.wav?raw=true" controls preload></audio> |
| funk | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/funk.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/35_funk.wav?raw=true" controls preload></audio> |
| gospel | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/gospel.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/40_gospel.wav?raw=true" controls preload></audio> |
| heavy metal | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/heavy_metal.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/41_heavy metal.wav?raw=true" controls preload></audio> |
| house | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/house.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/66_house.wav?raw=true" controls preload></audio> |
| indie rock | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/indie_rock.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/72_indie rock.wav?raw=true" controls preload></audio> |
| punk | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/punk.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/73_punk.wav?raw=true" controls preload></audio> |
| techno | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/techno.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/85_techno.wav?raw=true" controls preload></audio> |
| trance | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/trance.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/87_trance.wav?raw=true" controls preload></audio> |
| salsa | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/salsa.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/110_salsa.wav?raw=true" controls preload></audio> |
| samba | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/samba.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/113_samba.wav?raw=true" controls preload></audio> |
| soul | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/soul.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/00_soul.wav?raw=true" controls preload></audio> |
| swing | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/samples/swing.mp3?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/apnet/116_swing.wav?raw=true" controls preload></audio> |
|---|---|---|

#### Our models

|---|---|---|
|class | PECMAE-3 (prototype 0) | PECMAE-3 (prototype 1) |
|---|---|---|
| afrobeat | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_afrobeat_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_afrobeat_n1_gs1.wav?raw=true" controls preload> |
| ambient | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_ambient_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_ambient_n1_gs1.wav?raw=true" controls preload> |
| bolero | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_bolero_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_bolero_n1_gs1.wav?raw=true" controls preload> |
| bop | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_bop_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_bop_n1_gs1.wav?raw=true" controls preload> |
| bossa nova | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_bossa nova_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_bossa nova_n1_gs1.wav?raw=true" controls preload> |
| contemporary jazz | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_contemporary jazz_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_contemporary jazz_n1_gs1.wav?raw=true" controls preload> |
| cumbia | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_cumbia_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_cumbia_n1_gs1.wav?raw=true" controls preload> |
| disco | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_disco_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_disco_n1_gs1.wav?raw=true" controls preload> |
| doo wop | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_doo wop_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_doo wop_n1_gs1.wav?raw=true" controls preload> |
| dub | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_dub_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_dub_n1_gs1.wav?raw=true" controls preload> |
| electro | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_electro_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_electro_n1_gs1.wav?raw=true" controls preload> |
| europop | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_europop_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_europop_n1_gs1.wav?raw=true" controls preload> |
| funk | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_funk_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_funk_n1_gs1.wav?raw=true" controls preload> |
| gospel | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_gospel_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_gospel_n1_gs1.wav?raw=true" controls preload> |
| heavy metal | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_heavy metal_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_heavy metal_n1_gs1.wav?raw=true" controls preload> |
| house | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_house_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_house_n1_gs1.wav?raw=true" controls preload> |
| indie rock | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_indie rock_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_indie rock_n1_gs1.wav?raw=true" controls preload> |
| punk | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_punk_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_punk_n1_gs1.wav?raw=true" controls preload> |
| techno | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_techno_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_techno_n1_gs1.wav?raw=true" controls preload> |
| trance | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_trance_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_trance_n1_gs1.wav?raw=true" controls preload> |
| salsa | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_salsa_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_salsa_n1_gs1.wav?raw=true" controls preload> |
| samba | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_samba_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_samba_n1_gs1.wav?raw=true" controls preload> |
| soul | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_soul_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_soul_n1_gs1.wav?raw=true" controls preload> |
| swing | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_swing_n0_gs1.wav?raw=true" controls preload></audio> | <audio src="https://github.com/palonso/pecmae-samples/blob/main/xai_genre/pecmae-3/v492_swing_n1_gs1.wav?raw=true" controls preload> |
|---|---|---|

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
