#!/bin/bash
set -eu  # Crash if variable used without being set

# Setting environment
source /home/tdinh/.bashrc
conda activate fairseq
which python

# Where to store cache
export TORCH_HOME=/project/OML/tdinh/.cache/torch
export HF_HOME=/project/OML/tdinh/.cache/huggingface

export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID


KEPT_PORTION=90
DATA_DOMAIN="news"
COMET_REF_SYS="XCOMET-XL"
SELECTION_REF_SYS="XCOMET-XL"  # "BLEU"

SRC_LANG="en"
TGT_LANG="de"

#EXP_NAME="${DATA_DOMAIN}2022_XCOMET-XL"
EXP_NAME="rank_wmt22_official_submissions_${SRC_LANG}_${TGT_LANG}_mul_refs_paraphrased"
OUTPUT_DIR="output/${EXP_NAME}"

DATA_ROOT_PATH="/project/OML/tdinh/data"
MODEL_ROOT_PATH="/project/OML/tdinh/models"

declare -a QE_SYSTEMS=(  ) # "wmt22-cometkiwi-da" "wmt21-comet-qe-da" "wmt21-comet-qe-mqm" "wmt20-comet-qe-da" "wmt23-cometkiwi-da-xl" )
SYNTHETIC_QE_SYSTEMS="None"  # "20-human-with-noise"
QE_OFFICIAL_SUBMISSIONS="wmt22"

mkdir -p ${OUTPUT_DIR}

python process_data.py \
  --dataset "wmt22-testqe-mqm" \
  --data-root-path ${DATA_ROOT_PATH} \
  --src-lang ${SRC_LANG} \
  --tgt-lang ${TGT_LANG}

python paraphraser.py \
  --dataset "wmt22-testqe-mqm" \
  --data-root-path ${DATA_ROOT_PATH} \
  --src-lang ${SRC_LANG} \
  --tgt-lang ${TGT_LANG} \
  --model_name "Lelon/t5-german-paraphraser-large" \
  --nr_paraphrases 1 \
  --batch_size 64 \
  --temperature 0


if [[ ! -f ${OUTPUT_DIR}/human_based_rank.txt ]]; then
  python corr_eval_qe.py \
    --dataset "wmt22-testqe-mqm" \
    --data-root-path ${DATA_ROOT_PATH} \
    --data-domain ${DATA_DOMAIN} \
    --comet-model-root-path ${MODEL_ROOT_PATH} \
    --src-lang ${SRC_LANG} \
    --tgt-lang ${TGT_LANG} \
    --comet-qe-systems "${QE_SYSTEMS[@]}" \
    --synthetic-qe-systems ${SYNTHETIC_QE_SYSTEMS} \
    --qe-official-submissions ${QE_OFFICIAL_SUBMISSIONS} \
    --nr-gpus 1 \
    --batch-size 8 \
    --ground-truth "human" \
    --output ${OUTPUT_DIR}/human_based_rank.txt
fi

if [[ ! -f ${OUTPUT_DIR}/comet_ref_based_rank.txt ]]; then
  python corr_eval_qe.py \
    --dataset "wmt22-testqe-mqm" \
    --data-root-path ${DATA_ROOT_PATH} \
    --data-domain ${DATA_DOMAIN} \
    --comet-model-root-path ${MODEL_ROOT_PATH} \
    --src-lang ${SRC_LANG} \
    --tgt-lang ${TGT_LANG} \
    --comet-qe-systems "${QE_SYSTEMS[@]}" \
    --synthetic-qe-systems ${SYNTHETIC_QE_SYSTEMS} \
    --qe-official-submissions ${QE_OFFICIAL_SUBMISSIONS} \
    --nr-gpus 1 \
    --batch-size 8 \
    --ground-truth "ref_based_comet" \
    --ref-based-system ${COMET_REF_SYS} \
    --output ${OUTPUT_DIR}/comet_ref_based_rank.txt
fi

if [[ ! -f ${OUTPUT_DIR}/selection_based_rank_${SELECTION_REF_SYS}_${KEPT_PORTION}.txt ]]; then
  python selection_based_eval_qe.py \
    --dataset "wmt22-testqe-mqm" \
    --data-root-path ${DATA_ROOT_PATH} \
    --data-domain ${DATA_DOMAIN} \
    --comet-model-root-path ${MODEL_ROOT_PATH} \
    --src-lang ${SRC_LANG} \
    --tgt-lang ${TGT_LANG} \
    --comet-qe-systems "${QE_SYSTEMS[@]}" \
    --synthetic-qe-systems ${SYNTHETIC_QE_SYSTEMS} \
    --qe-official-submissions ${QE_OFFICIAL_SUBMISSIONS} \
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
    --candidate-rankers "comet_ref_based_rank" "selection_based_rank_XCOMET-XL_90" \
    --eval-ranker-output-path ${OUTPUT_DIR}/meta_meta_eval_qe.txt
fi

# "comet_ref_based_rank" "selection_based_rank_BLEU_90" "selection_based_rank_CHRF_90" "selection_based_rank_XCOMET-XL_70" "selection_based_rank_XCOMET-XL_80" "selection_based_rank_XCOMET-XL_90" "selection_based_rank_XCOMET-XL_95" \