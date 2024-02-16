#!/bin/bash
set -eu  # Crash if variable used without being set

# Setting environment
source /home/tdinh/.bashrc
conda activate fairseq
which python

# Where to store cache
export TORCH_HOME=/project/OML/tdinh/.cache/torch
export HF_HOME=/project/OML/tdinh/.cache/huggingface

export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID


KEPT_PORTION=90
DATASET="wmt22-testqe-mqm"
DATA_DOMAIN="news"
REF_BASED_SYS="BLEU" # 'XCOMET-XL', 'wmt22-comet-da', 'BLEU', 'CHRF', 'TER'

SRC_LANG="en"
TGT_LANG="de"

declare -a QE_SYSTEMS=(  )  # "wmt22-cometkiwi-da" "wmt21-comet-qe-da" "wmt21-comet-qe-mqm" "wmt20-comet-qe-da" "wmt23-cometkiwi-da-xl" )
SYNTHETIC_QE_SYSTEMS="None"  # "20-human-with-noise"
QE_OFFICIAL_SUBMISSIONS="wmt22"  # "wmt22"

EXP_NAME="rank_wmt22_official_submissions_${SRC_LANG}_${TGT_LANG}_${REF_BASED_SYS}_mul_refs" # _paraphrased"
OUTPUT_DIR="output/${EXP_NAME}"

DATA_ROOT_PATH="/project/OML/tdinh/data"
MODEL_ROOT_PATH="/project/OML/tdinh/models"

mkdir -p ${OUTPUT_DIR}

#python process_data.py \
#  --dataset ${DATASET} \
#  --data-root-path ${DATA_ROOT_PATH} \
#  --src-lang ${SRC_LANG} \
#  --tgt-lang ${TGT_LANG}

#python paraphraser.py \
#  --dataset ${DATASET} \
#  --data-root-path ${DATA_ROOT_PATH} \
#  --src-lang ${SRC_LANG} \
#  --tgt-lang ${TGT_LANG} \
#  --model_name "Lelon/t5-german-paraphraser-large" \
#  --nr_paraphrases 3 \
#  --batch_size 64 \
#  --temperature 1


if [[ ! -f ${OUTPUT_DIR}/human_based_rank.txt ]]; then
  python corr_eval_qe.py \
    --dataset ${DATASET} \
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

if [[ ! -f ${OUTPUT_DIR}/ref_based_rank.txt ]]; then
  python corr_eval_qe.py \
    --dataset ${DATASET} \
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
    --ground-truth "ref_based_metrics" \
    --ref-based-system ${REF_BASED_SYS} \
    --output ${OUTPUT_DIR}/ref_based_rank.txt
fi

#if [[ ! -f ${OUTPUT_DIR}/selection_based_rank_${REF_BASED_SYS}_${KEPT_PORTION}.txt ]]; then
#  python selection_based_eval_qe.py \
#    --dataset ${DATASET} \
#    --data-root-path ${DATA_ROOT_PATH} \
#    --data-domain ${DATA_DOMAIN} \
#    --comet-model-root-path ${MODEL_ROOT_PATH} \
#    --src-lang ${SRC_LANG} \
#    --tgt-lang ${TGT_LANG} \
#    --comet-qe-systems "${QE_SYSTEMS[@]}" \
#    --synthetic-qe-systems ${SYNTHETIC_QE_SYSTEMS} \
#    --qe-official-submissions ${QE_OFFICIAL_SUBMISSIONS} \
#    --nr-gpus 1 \
#    --batch-size 8 \
#    --kept-portion ${KEPT_PORTION} \
#    --ref-based-system ${REF_BASED_SYS} \
#    --output ${OUTPUT_DIR}/selection_based_rank_${REF_BASED_SYS}_${KEPT_PORTION}.txt
#fi

if [[ ! -f ${OUTPUT_DIR}/meta_meta_eval_qe.txt ]]; then
  python meta_meta_eval_qe.py \
    --output-dir ${OUTPUT_DIR} \
    --gold-ranker "human_based_rank" \
    --candidate-rankers "ref_based_rank" \
    --eval-ranker-output-path ${OUTPUT_DIR}/meta_meta_eval_qe.txt
fi

# "selection_based_rank_XCOMET-XL_90"
# "ref_based_rank" "selection_based_rank_BLEU_90" "selection_based_rank_CHRF_90" "selection_based_rank_XCOMET-XL_70" "selection_based_rank_XCOMET-XL_80" "selection_based_rank_XCOMET-XL_90" "selection_based_rank_XCOMET-XL_95" \