#!/bin/bash
# Stage-2 nominal demo generation: 8 seeded runs x 63 eps = 504 episodes.
# Seeds vary the per-ENV object scale (pinned per run) + spawn stream; --diversify
# adds order shuffle + candidate dropout (PLAN 2.2). Sequential = one GPU job.
# Usage: nohup bash gen_demos_nominal.sh > gen_demos_nominal.log 2>&1 &
set -u
cd "$(dirname "$0")"
source /home/william/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
for SEED in 101 102 103 104 105 106 107 108; do
    OUT="demos_nominal_s${SEED}.h5"
    if [ -f "$OUT" ]; then
        echo "[gen] $OUT exists, skipping seed $SEED"
        continue
    fi
    echo "[gen] seed $SEED starting $(date -Is)"
    python run_expert_v1.py --episodes 63 --seed "$SEED" --diversify \
        --record-h5 "$OUT" > "gen_nominal_s${SEED}.log" 2>&1
    RC=$?
    # the runner writes fixed-name artifacts — snapshot them per seed
    [ -f expert_v1_results.json ] && cp expert_v1_results.json "results_nominal_s${SEED}.json"
    echo "[gen] seed $SEED done rc=$RC $(date -Is)"
    if [ $RC -ne 0 ]; then
        echo "[gen] seed $SEED FAILED (rc=$RC) — aborting chain"
        exit $RC
    fi
done
echo "[gen] ALL SEEDS DONE $(date -Is)"
