#!/bin/bash
# QE-M Task on WMT24: rank the reference-free submissions to the WMT24 metrics shared task with a reference-based
# metric, and compare with the ranking by human ESA scores. Same steps as run.sh, single reference (refA), no
# paraphrases; the data is read directly from mt-metrics-eval (no process_data.py step).
#   bash run_wmt24.sh <ref_based_metric> <lang_pair> <domain> <gpu>
# e.g. bash run_wmt24.sh XCOMET-XL en-cs all 1. Domains: all; en-*: literary, news, social, speech;
# cs-uk: education, news, official, personal, voice.
set -eu  # Crash if variable used without being set

# Setting environment
source /home/tdinh/.bashrc
conda activate fairseq
which python

# Where to store cache
export TORCH_HOME=/project/t80p2/tdinh/.cache/torch
export HF_HOME=/project/t80p2/tdinh/.cache/huggingface

export CUDA_VISIBLE_DEVICES=$4
export CUDA_DEVICE_ORDER=PCI_BUS_ID

DATASET="wmt24-testmetrics-esa"
QE_OFFICIAL_SUBMISSIONS="wmt24-qe-metrics"
REF_BASED_SYS=$1  # 'XCOMET-XL' 'wmt22-comet-da' 'BLEU' 'CHRF' 'TER' 'bert-score' 'bleurt' 'unite-mup' "metricx-23-xl-v2p0"
LANG_PAIR=$2
DATA_DOMAIN=$3
SRC_LANG=${LANG_PAIR%-*}
TGT_LANG=${LANG_PAIR#*-}

if [[ ${REF_BASED_SYS} == 'bleurt' ]]; then
  . .venv/bin/activate
elif [[ ${REF_BASED_SYS} == 'metricx-23-xl-v2p0' ]]; then
  . .metricx/bin/activate
fi

EXP_NAME="rank_${QE_OFFICIAL_SUBMISSIONS}_${LANG_PAIR}_${DATA_DOMAIN}_${REF_BASED_SYS}_single_0"
OUTPUT_DIR="output/${EXP_NAME}"
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}

DATA_ROOT_PATH="/project/t80p2/tdinh/data"
MODEL_ROOT_PATH="/project/t80p2/tdinh/models"
REF_BASED_CACHE_DIR="cache/ref_based_scores"  # model-based metrics' segment scores, shared across runs

WANDB_USERNAME="tuanh"  # Replace with your username if use wandb, otherwise "None"
WANDB_PROJECT="auto-meta-eval-qe"
WANDB_RUN_ID=$(date +%s%N)

echo ${WANDB_RUN_ID} > ${OUTPUT_DIR}/wandb_run_id.txt

for GROUND_TRUTH in "human" "ref_based_metrics"; do
  python corr_eval_qe.py \
    --dataset ${DATASET} \
    --data-root-path ${DATA_ROOT_PATH} \
    --data-domain ${DATA_DOMAIN} \
    --comet-model-root-path ${MODEL_ROOT_PATH} \
    --src-lang ${SRC_LANG} \
    --tgt-lang ${TGT_LANG} \
    --qe-official-submissions ${QE_OFFICIAL_SUBMISSIONS} \
    --nr-gpus 1 \
    --batch-size 8 \
    --ground-truth ${GROUND_TRUTH} \
    --ref-based-cache-dir ${REF_BASED_CACHE_DIR} \
    --ref-based-system ${REF_BASED_SYS} \
    --wandb-username ${WANDB_USERNAME} \
    --wandb-run-id ${WANDB_RUN_ID} \
    --wandb-project ${WANDB_PROJECT} \
    --output-dir ${OUTPUT_DIR}
done

python meta_meta_eval_qe.py \
  --output-dir ${OUTPUT_DIR} \
  --gold-ranker "human_based_rank" \
  --candidate-ranker "ref_based_rank" \
  --wandb-username ${WANDB_USERNAME} \
  --wandb-run-id ${WANDB_RUN_ID} \
  --wandb-project ${WANDB_PROJECT} \
  --eval-ranker-output-path ${OUTPUT_DIR}/meta_meta_eval_qe.txt

if [[ ( ${REF_BASED_SYS} == 'bleurt' ) || ( ${REF_BASED_SYS} == 'metricx-23-xl-v2p0' ) ]]; then
  deactivate
fi

mv ${OUTPUT_DIR} ${OUTPUT_DIR}_${WANDB_RUN_ID}
