#!/bin/bash
# Sync between this machine (where the code is written and version-controlled) and the ISL servers. trg3 mounts
# /project/t80p2, so rsync goes straight to trg3; the GPU servers (e.g. 63) see the same files.
#   bash sync.sh push   # code (git-tracked + untracked non-ignored files)
#   bash sync.sh pull   # output/ back from the cluster
set -euo pipefail
REMOTE=trg3
REMOTE_DIR=/project/t80p2/tdinh/auto-meta-eval-qe
cd "$(dirname "$0")"

case "${1:-}" in
  push)
    git rev-parse --short HEAD > commit.txt && git diff > diff.txt  # which version runs remotely
    rsync -ahr --stats ./ "$REMOTE:$REMOTE_DIR" --exclude='/.git' --exclude='/output' --exclude='/wandb' \
        --exclude='/.venv' --exclude='/.metricx' --exclude='/logs' --filter=':- .gitignore' ;;
  pull) rsync -ahr --stats "$REMOTE:$REMOTE_DIR/output/" ./output/ ;;
  *) echo "usage: $0 push|pull"; exit 1 ;;
esac
