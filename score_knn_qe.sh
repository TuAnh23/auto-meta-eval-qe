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

declare -a DATASETS=( 'ted_tst14.en' 'wmt22general.en' )
declare -a MT_MODELS=( 'ted_new' 'deltalm_base_ft_ted' )
declare -a EMB_LAYERS=( 6 )
declare -a DATASTORES=( 'europarlshort' 'reduced_ted_01' 'reduced_ted_02' 'reduced_ted_05' 'reduced_ted_10' 'reduced_ted_20'  'reduced_ted_40' 'reduced_ted_60' 'reduced_ted_80' 'ted_new' )

for dataset in "${DATASETS[@]}"; do
  for mt_model in "${MT_MODELS[@]}"; do
    for emb_layer in "${EMB_LAYERS[@]}"; do
      for datastore in "${DATASTORES[@]}"; do
        if [[ ${dataset} == 'wmt22general.en' ]]; then
          declare -a DOMAINS=( 'all' 'conversation' 'ecommerce' 'news' 'social' )
        else
          declare -a DOMAINS=( 'None' )
        fi
        for domain in "${DOMAINS[@]}"; do
          python score_knn_qe.py \
            --dataset ${dataset} \
            --mt_model ${mt_model} \
            --emb_layer ${emb_layer} \
            --datastore ${datastore} \
            --domain ${domain} \
            --data_root_path '/project/OML/tdinh/data' \
            --knn_qe_root_path '/project/OML/tdinh/knn-qe'
        done
      done
    done
  done
done