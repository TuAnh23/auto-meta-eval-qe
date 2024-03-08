#!/bin/bash
set -eu  # Crash if variable used without being set

# Setting environment
source /home/tdinh/.bashrc
conda activate fairseq
. .metricx/bin/activate
which python

# Where to store cache
export TORCH_HOME=/project/OML/tdinh/.cache/torch
export HF_HOME=/project/OML/tdinh/.cache/huggingface

export CUDA_VISIBLE_DEVICES=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID

python score_knn_qe.py \
  --dataset 'ted_tst14.en' \
  --mt_model 'ted_new' \
  --emb_layer 6 \
  --datastore 'ted_new' \
  --domain 'None' \
  --data_root_path '/project/OML/tdinh/data' \
  --knn_qe_root_path '/project/OML/tdinh/knn-qe'
