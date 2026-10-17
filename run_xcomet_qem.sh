#!/bin/bash
# XCOMET-XL runs on the WMT22 QE-as-a-Metric task (QE-M) that are missing for the EAMT 2024 paper (Table 2, Fig. 2-5).
#   bash run_xcomet_qem.sh <gpu_a> <gpu_b> <gpu_c>     # in tmux on the GPU server; logs in logs/xcomet/
# Runs on the same domain share data/wmt22_general_MT/en-de, which process_data.py and paraphraser.py rewrite at the
# start of every run. So each 'all'-domain run starts only after the previous one has loaded its data, i.e. after the
# ref-based pass of corr_eval_qe.py printed "MT systems to be scored" (the 2nd time in the log).
# XCOMET-XL segment scores are cached (cache/ref_based_scores/XCOMET-XL), so later runs reuse earlier runs' scores.
set -u
cd "$(dirname "$0")"
export WANDB_MODE=offline PYTHONUNBUFFERED=1
mkdir -p logs/xcomet

run() {  # <single|mul> <nr_paraphrases> <gpu> <domain>
  local log="logs/xcomet/${4}_${1}_${2}.log"
  bash run.sh XCOMET-XL "$1" "$2" "$3" "$4" > "$log" 2>&1 && echo "$(date) done   $4 $1 $2" || echo "$(date) FAILED $4 $1 $2 (see $log)"
}

wait_data_loaded() {  # <log> <pid>
  until [ "$(grep -c 'MT systems to be scored' "$1" 2>/dev/null)" -ge 2 ]; do
    kill -0 "$2" 2>/dev/null || { echo "$(date) $1 ended before loading its data"; return; }
    sleep 30
  done
}

echo "$(date) start all/mul/2 (6 refs) on GPU $1"
run mul 2 "$1" all & pid_a=$!
wait_data_loaded logs/xcomet/all_mul_2.log $pid_a

echo "$(date) start all/mul/1 (4 refs) on GPU $2"
run mul 1 "$2" all & pid_b=$!
wait_data_loaded logs/xcomet/all_mul_1.log $pid_b

echo "$(date) start chain of the remaining runs on GPU $3"
(
  run mul 0 "$3" all
  run single 0 "$3" all
  for domain in news conversation ecommerce social; do  # news: XCOMET-XL runs exist from 2024, rerun as a check
    run mul 0 "$3" $domain
    run single 0 "$3" $domain
  done
) & pid_c=$!

wait $pid_a $pid_b $pid_c
echo "$(date) ALL DONE"
