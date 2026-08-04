#!/usr/bin/env bash
# Re-run the two widened-spawn cells on the binary that records spawn_yaw (EXP_ROBUSTNESS §6e).
#
# §6e had to leave one thing unresolved: at --spawn-scale 2.0 even the cohort whose spawn
# POSITION fell inside the nominal box scored 0.783 against the gate's 0.979. The leading
# explanation is that --spawn-scale widens the YAW range too (+/-0.35 -> +/-0.70 rad) and
# spawn_pos records only position, so "inside the box" silently mixed in-distribution positions
# with out-of-distribution yaws. eval_act.py now records spawn_yaw (added 09:27:43), which turns
# that from a re-run into a read-back -- but only for cells collected after the change.
#
# These two cells carry no new perturbation: same checkpoint, same seed, same flags as spawn15 /
# spawn20. So they are also a free REPRODUCIBILITY check -- yaw_of() consumes no RNG, so the
# outcomes should be episode-for-episode identical to the originals. If they are not, something
# in the pipeline is not as deterministic as every comparison in this document assumes.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

RUN="${1:-runs/bc_armB_seed0}"
for CELL in "spawn15yaw:1.5" "spawn20yaw:2.0"; do
  NAME="${CELL%%:*}"; SCALE="${CELL##*:}"
  OUT="$RUN/robust_${NAME}.json"
  [ -f "$OUT" ] && { echo "SKIP $OUT"; continue; }
  echo "=== $NAME  --spawn-scale $SCALE  $(date -Is) ==="
  python slot_act/eval_act.py --ckpt "$RUN/ckpt_final.pt" --task Rebot-PrecisionSlot-v0 \
    --num-envs 32 --episodes 128 --seed 777 --spawn-scale "$SCALE" --out "$OUT" 2>&1 \
    | grep -E "eval_act" || echo "!!! FAILED (continuing): $OUT"
done
echo "=== spawn-yaw cells done  $(date -Is) ==="
