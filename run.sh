#!/bin/bash
set -eu  # Crash if variable used without being set

# Setting environment
source /home/tdinh/.bashrc
conda activate fairseq
which python

# Where to store cache
export TORCH_HOME=/project/OML/tdinh/.cache/torch
export HF_HOME=/project/OML/tdinh/.cache/huggingface

export CUDA_VISIBLE_DEVICES=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID


KEPT_PORTION=90
DATA_DOMAIN="news"
COMET_REF_SYS="XCOMET-XL"
SELECTION_REF_SYS=${COMET_REF_SYS}

EXP_NAME="${DATA_DOMAIN}2022_XCOMET-XL"
OUTPUT_DIR="output/${EXP_NAME}"

DATA_ROOT_PATH="/project/OML/tdinh/data"
MODEL_ROOT_PATH="/project/OML/tdinh/models"

declare -a QE_SYSTEMS=( "wmt22-cometkiwi-da" "wmt21-comet-qe-da" "wmt21-comet-qe-mqm" "wmt20-comet-qe-da" "wmt23-cometkiwi-da-xl" )

mkdir -p ${OUTPUT_DIR}

if [[ ! -f ${OUTPUT_DIR}/human_based_rank.txt ]]; then
  python corr_eval_qe.py \
    --dataset "metrics_wmt-mqm-human-evaluation" \
    --data-root-path ${DATA_ROOT_PATH} \
    --data-domain ${DATA_DOMAIN} \
    --comet-model-root-path ${MODEL_ROOT_PATH} \
    --src-lang "en" \
    --tgt-lang "de" \
    --qe-systems "${QE_SYSTEMS[@]}" \
    --nr-gpus 1 \
    --batch-size 8 \
    --ground-truth "human" \
    --output ${OUTPUT_DIR}/human_based_rank.txt
fi

if [[ ! -f ${OUTPUT_DIR}/comet_ref_based_rank.txt ]]; then
  python corr_eval_qe.py \
    --dataset "metrics_wmt-mqm-human-evaluation" \
    --data-root-path ${DATA_ROOT_PATH} \
    --data-domain ${DATA_DOMAIN} \
    --comet-model-root-path ${MODEL_ROOT_PATH} \
    --src-lang "en" \
    --tgt-lang "de" \
    --qe-systems "${QE_SYSTEMS[@]}" \
    --nr-gpus 1 \
    --batch-size 8 \
    --ground-truth "ref_based_comet" \
    --ref-based-system ${COMET_REF_SYS} \
    --output ${OUTPUT_DIR}/comet_ref_based_rank.txt
fi

if [[ ! -f ${OUTPUT_DIR}/selection_based_rank_${SELECTION_REF_SYS}_${KEPT_PORTION}.txt ]]; then
  python selection_based_eval_qe.py \
    --dataset "metrics_wmt-mqm-human-evaluation" \
    --data-root-path ${DATA_ROOT_PATH} \
    --data-domain ${DATA_DOMAIN} \
    --comet-model-root-path ${MODEL_ROOT_PATH} \
    --src-lang "en" \
    --tgt-lang "de" \
    --qe-systems "${QE_SYSTEMS[@]}" \
    --nr-gpus 1 \
    --batch-size 8 \
    --kept-portion ${KEPT_PORTION} \
    --ref-based-system ${SELECTION_REF_SYS} \
    --output ${OUTPUT_DIR}/selection_based_rank_${SELECTION_REF_SYS}_${KEPT_PORTION}.txt
fi

if [[ ! -f ${OUTPUT_DIR}/meta_meta_eval_qe.txt ]]; then
  python meta_meta_eval_qe.py \
    --output-dir ${OUTPUT_DIR} \
    --gold-ranker "human_based_rank" \
    --candidate-rankers "comet_ref_based_rank" "selection_based_rank_BLEU_90" "selection_based_rank_CHRF_90" "selection_based_rank_XCOMET-XL_70" "selection_based_rank_XCOMET-XL_80" "selection_based_rank_XCOMET-XL_90" "selection_based_rank_XCOMET-XL_95" \
    --eval-ranker-output-path ${OUTPUT_DIR}/meta_meta_eval_qe.txt
fi