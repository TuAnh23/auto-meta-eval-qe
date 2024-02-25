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

DATASET="wmt22-testmetrics"
DATA_DOMAIN="all"  # "news" "all"
REF_BASED_SYS="TER" # 'XCOMET-XL', 'wmt22-comet-da', 'BLEU', 'CHRF', 'TER', 'bert-score'
NR_ORIGINAL_REFERENCE="mul"  # "single", "mul"

SRC_LANG="en"
TGT_LANG="de"

declare -a QE_SYSTEMS=(  ) # "wmt22-cometkiwi-da" "wmt21-comet-qe-da" "wmt21-comet-qe-mqm" "wmt20-comet-qe-da" "wmt23-cometkiwi-da-xl" )
SYNTHETIC_QE_SYSTEMS="None"  # "20-human-with-noise"
QE_OFFICIAL_SUBMISSIONS="wmt22-qe-metrics"  # "wmt22" "None" "wmt22-testmetrics"

EXP_NAME="rank_wmt22_official_submissions"
OUTPUT_DIR="output/${EXP_NAME}"
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}

DATA_ROOT_PATH="/project/OML/tdinh/data"
MODEL_ROOT_PATH="/project/OML/tdinh/models"

WANDB_USERNAME="tuanh"  # Replace with your username if use wandb, otherwise "None"
WANDB_PROJECT="auto-meta-eval-qe"
WANDB_RUN_ID=$(date +%s%N)

echo ${WANDB_RUN_ID} > ${OUTPUT_DIR}/wandb_run_id.txt
#WANDB_RUN_ID=$(cat ${OUTPUT_DIR}/wandb_run_id.txt)

python process_data.py \
  --dataset ${DATASET} \
  --data-root-path ${DATA_ROOT_PATH} \
  --src-lang ${SRC_LANG} \
  --tgt-lang ${TGT_LANG} \
  --nr-refs ${NR_ORIGINAL_REFERENCE} \
  --wandb-username ${WANDB_USERNAME} \
  --wandb-run-id ${WANDB_RUN_ID} \
  --wandb-project ${WANDB_PROJECT}

python paraphraser.py \
  --dataset ${DATASET} \
  --data-root-path ${DATA_ROOT_PATH} \
  --src-lang ${SRC_LANG} \
  --tgt-lang ${TGT_LANG} \
  --paraphraser_name "Lelon/t5-german-paraphraser-large" \
  --nr_paraphrases 1 \
  --batch_size 64 \
  --temperature 0 \
  --wandb-username ${WANDB_USERNAME} \
  --wandb-run-id ${WANDB_RUN_ID} \
  --wandb-project ${WANDB_PROJECT}


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
    --ref-based-system ${REF_BASED_SYS} \
    --wandb-username ${WANDB_USERNAME} \
    --wandb-run-id ${WANDB_RUN_ID} \
    --wandb-project ${WANDB_PROJECT} \
    --output-dir ${OUTPUT_DIR}
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
    --wandb-username ${WANDB_USERNAME} \
    --wandb-run-id ${WANDB_RUN_ID} \
    --wandb-project ${WANDB_PROJECT} \
    --output-dir ${OUTPUT_DIR}
fi

if [[ ! -f ${OUTPUT_DIR}/meta_meta_eval_qe.txt ]]; then
  python meta_meta_eval_qe.py \
    --output-dir ${OUTPUT_DIR} \
    --gold-ranker "human_based_rank" \
    --candidate-ranker "ref_based_rank" \
    --wandb-username ${WANDB_USERNAME} \
    --wandb-run-id ${WANDB_RUN_ID} \
    --wandb-project ${WANDB_PROJECT} \
    --eval-ranker-output-path ${OUTPUT_DIR}/meta_meta_eval_qe.txt
fi

mv ${OUTPUT_DIR} ${OUTPUT_DIR}_${WANDB_RUN_ID}