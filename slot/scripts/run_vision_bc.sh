#!/usr/bin/env bash
# VISION_PLAN G3 + G4 — the blind control FIRST, then the visual policy.
#
# Order is deliberate and not negotiable. The slot never moves, so a blind policy can execute
# the entire insertion; it just cannot find the block. If blind scores >= 0.60 then the eval
# barely tests vision and a good visual number means nothing until the spawn box is widened.
# Running blind first means that verdict exists BEFORE there is a visual number to be excited
# about, which is the only ordering that keeps me honest.
#
# Both arms: identical architecture, identical capacity, identical step budget, identical data.
# The single difference is that the blind arm's images are zeroed -- at train AND eval, enforced
# by an assert in eval_vision.py, because a blind checkpoint evaluated sighted is the same
# train/test mismatch that scored EXP08's student 0.0%/1.6%.
#
# Eval seeds 777/888 were never collected on (collection ran 101/202) -- the student is scored
# on spawns it has never seen.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

DATA=(data/vision_bc/seed101 data/vision_bc/seed202)
STEPS="${STEPS:-60000}"
RUNS=runs/vision_bc

train () {  # name, extra flags...
  local name="$1"; shift
  [ -f "$RUNS/$name/ckpt_final.pt" ] && { echo "SKIP train $name"; return 0; }
  echo "=== train $name  ($STEPS steps)  $(date -Is) ==="
  python slot_act/train_flow_vision.py --data "${DATA[@]}" --out "$RUNS/$name" \
    --steps "$STEPS" --seed 0 "$@"
}

evaluate () {  # name, seed, extra flags...
  local name="$1" seed="$2"; shift 2
  local out="$RUNS/$name/eval_s$seed.json"
  [ -f "$out" ] && { echo "SKIP eval $out"; return 0; }
  echo "=== eval $name seed $seed  $(date -Is) ==="
  python scripts/eval_vision.py --ckpt "$RUNS/$name/ckpt_final.pt" --episodes 128 \
    --num-envs 16 --seed "$seed" --out "$out" "$@" 2>&1 | grep -E "eval_vision|render contract"
}

# --- G3: the control. Same net, no information. ---
train    blind --blind
evaluate blind 777 --blind
evaluate blind 888 --blind

# --- G4: the visual policy. ---
train    v1
evaluate v1 777
evaluate v1 888

echo
echo "=== vision BC done  $(date -Is) ==="
python - <<'PY'
import json, glob
print(f"{'run':<8}{'seed':>6}{'later':>9}{'n':>5}")
for f in sorted(glob.glob('runs/vision_bc/*/eval_s*.json')):
    d = json.load(open(f))
    run = f.split('/')[2]
    print(f"{run:<8}{d['seed']:>6}{d['success_rate_later']:>9.3f}{d['n_later']:>5}")
print("\nchampion (privileged state, seed 777): 0.979   |   bar: 0.80")
print("read the visual number ONLY against the blind number on the same seed")
PY
