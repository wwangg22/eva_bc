#!/usr/bin/env bash
# VISION_PLAN G5 — retrain on BC + DAgger, from scratch, then evaluate.
#
# From scratch on the COMBINED pool rather than fine-tuning the v1 checkpoint, matching EXP08's
# Gate D. Fine-tuning would leave open whether the result inherits v1's data distribution, and
# with only ~1.8 h of training at stake the cleaner experiment is worth it. Same 60k steps, same
# recipe, same eval seeds as v1, so v2 - v1 is attributable to the data and nothing else.
#
# KNOWN IMBALANCE, stated before the number arrives: the BC pool contributes ~145,800 samples
# (every timestep of 243 episodes) and the DAgger pool ~10,240 (40 window boundaries x 256
# episodes), so DAgger states are only **6.6 %** of the mix. That is the straightforward
# concatenation EXP08 used. If v2 comes back flat against v1, oversampling the DAgger shards is
# the first thing to try -- NOT the conclusion that DAgger does not help here.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

DATA=(data/vision_bc/seed101 data/vision_bc/seed202 data/vision_dagger/seed303 data/vision_dagger/seed404)
STEPS="${STEPS:-60000}"
RUN=runs/vision_bc/v2_dagger

if [ ! -f "$RUN/ckpt_final.pt" ]; then
  echo "=== train v2_dagger ($STEPS steps)  $(date -Is) ==="
  python slot_act/train_flow_vision.py --data "${DATA[@]}" --out "$RUN" --steps "$STEPS" --seed 0
else
  echo "SKIP train (ckpt exists)"
fi

for SEED in 777 888; do
  OUT="$RUN/eval_s$SEED.json"
  [ -f "$OUT" ] && { echo "SKIP eval $OUT"; continue; }
  echo "=== eval v2_dagger seed $SEED  $(date -Is) ==="
  python scripts/eval_vision.py --ckpt "$RUN/ckpt_final.pt" --episodes 128 --num-envs 16 \
    --seed "$SEED" --out "$OUT" 2>&1 \
    | grep -E "eval_vision|render contract|Traceback|Error|error|assert" || true
  [ -f "$OUT" ] || echo "!!! EVAL PRODUCED NO OUTPUT: $OUT"
done

echo
echo "=== v2 done  $(date -Is) ==="
python - <<'PY'
import json, glob, sys
sys.path.insert(0, 'analysis')
from robustness_report import wilson, two_prop_z
rows = {}
for arm in ('blind', 'v1', 'v2_dagger'):
    K = N = 0
    for f in sorted(glob.glob(f'runs/vision_bc/{arm}/eval_s*.json')):
        d = json.load(open(f))
        l = [e for e in d['per_episode'] if e['episode_index_in_env'] > 0]
        K += sum(e['success'] for e in l); N += len(l)
    if N:
        rows[arm] = (K, N)
        lo, hi = wilson(K, N)
        print(f"  {arm:<10} {K}/{N} = {K/N:.3f}  Wilson [{lo:.3f}, {hi:.3f}]")
if 'v1' in rows and 'v2_dagger' in rows:
    (k2, n2), (k1, n1) = rows['v2_dagger'], rows['v1']
    z, p = two_prop_z(k2, n2, k1, n1)
    print(f"\n  DAgger effect (v2 - v1): {100*(k2/n2 - k1/n1):+.1f} pts   z={z:+.2f}  p={p:.4f}")
    print(f"  >= 0.80 bar: {'CLEARED' if k2/n2 >= 0.80 else 'not cleared'}"
          f"   Wilson low {wilson(k2,n2)[0]:.3f}")
PY
