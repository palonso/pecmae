#!/bin/bash

set -e
cd "$(dirname "$0")"/../

python src/train_protos.py \
    --data-dir feats/gtzan/diffusion_4s/ \
    --dataset gtzan \
    --metadata-file-train groundtruth/gtzan/groundtruth_train.tsv \
    --metadata-file-val groundtruth/gtzan/groundtruth_val.tsv \
    --metadata-file-test groundtruth/gtzan/groundtruth_test.tsv \
    --protos-init kmeans-centers \
    --n-protos-per-label 5 \
    --batch-size 256 \
    --alpha 0.75 \
    --proto-loss-samples class \
    --proto-loss l2 \
    --total-steps 150000 \
    --timestamps 1 \
    --max-lr 1e-3 \
    --gpu-id 0 \
    --trim-mode all \
    --time-summarization transformer \
    --use-discriminator False \
    --do-normalization True
