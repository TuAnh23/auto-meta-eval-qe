#!/bin/bash
# All WMT24 ESA runs (9 language pairs x 9 ref-based metrics x (all + each domain)), in 4 GPU workers + 1 CPU worker.
#   bash run_wmt24_batch.sh <gpu_a> <gpu_b> <gpu_c> <gpu_d>     # in tmux on the GPU server; logs in logs/wmt24/
# Each worker runs its jobs (<metric>:<lang pairs>) one after the other; per language pair the 'all' run comes first and
# scores every segment, the domain runs after it read their scores from the cache (cache/ref_based_scores).
# XCOMET-XL and MetricX-23 XL (~3 segments/s on a TITAN RTX) are split over two workers each, by language pair; TER
# needs no GPU but is slow on zh/ja (character-level TER on long segments), so it gets its own worker.
# Runs that already have an output folder (with meta_meta_eval_qe.txt) are skipped, so the batch can be restarted.
set -u
cd "$(dirname "$0")"
export WANDB_MODE=offline PYTHONUNBUFFERED=1
mkdir -p logs/wmt24

ALL_PAIRS="en-cs,en-es,en-hi,en-is,en-ja,en-ru,en-uk,en-zh,cs-uk"
PAIRS_A="cs-uk,en-cs,en-es"  # ~38k segments with ESA scores
PAIRS_B="en-hi,en-is,en-ja,en-ru,en-uk,en-zh"  # ~43k

domains() {  # <lang_pair>
  if [[ $1 == cs-uk ]]; then echo "all education news official personal voice"; else echo "all literary news social speech"; fi
}

worker() {  # <gpu> <metric>:<comma-separated lang pairs>...
  local gpu=$1; shift
  for job in "$@"; do
    local metric=${job%%:*}
    for lp in $(echo "${job#*:}" | tr , " "); do
      for domain in $(domains $lp); do
        local name="rank_wmt24-qe-metrics_${lp}_${domain}_${metric}_single_0"
        if ls output/${name}_*/meta_meta_eval_qe.txt > /dev/null 2>&1; then
          echo "$(date) skip   $lp $domain $metric (done)"; continue
        fi
        local log="logs/wmt24/${lp}_${domain}_${metric}.log"
        bash run_wmt24.sh "$metric" "$lp" "$domain" "$gpu" > "$log" 2>&1 \
          && echo "$(date) done   $lp $domain $metric" || echo "$(date) FAILED $lp $domain $metric (see $log)"
      done
    done
  done
}

worker "$1" XCOMET-XL:$PAIRS_A & p1=$!
worker "$2" metricx-23-xl-v2p0:$PAIRS_A & p2=$!
worker "$3" bleurt:$ALL_PAIRS bert-score:$ALL_PAIRS BLEU:$ALL_PAIRS XCOMET-XL:$PAIRS_B & p3=$!
worker "$4" wmt22-comet-da:$ALL_PAIRS unite-mup:$ALL_PAIRS CHRF:$ALL_PAIRS metricx-23-xl-v2p0:$PAIRS_B & p4=$!
worker "$4" TER:$ALL_PAIRS & p5=$!  # CPU only
wait $p1 $p2 $p3 $p4 $p5
echo "$(date) ALL DONE"
