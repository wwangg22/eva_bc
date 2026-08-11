#!/usr/bin/env bash
# The deploy-readiness robustness matrix (05_VISUAL_DR.md phase 6).
#
# Evaluates ONE checkpoint per-axis on the -VisionDR task: every row turns a single DR
# group on (all others off) at a given magnitude, plus a nominal row (all off) and an
# all-on row at the training magnitudes and at 1.5x (held-out severity). Success-vs-axis
# is the direct answer to "how will this deploy", and a collapsed row names the axis to
# feed back into the next DR round.
#
#   bash re3sim/act/eval_vdr_matrix.sh <ckpt> <outdir> [num_envs] [seeds]
#
# Serial by design: ONE Isaac instance at a time on this machine (05_VISUAL_DR.md 4b).
set -euo pipefail

CKPT=$(readlink -f "${1:?checkpoint}")   # absolute BEFORE the cd below
OUT=$(readlink -f -m "${2:?output dir}")
N=${3:-32}
SEEDS=${4:-88000,88001}
# -Play spacing (2.0 m) so the nominal row is the SAME visual protocol earlier
# checkpoints were evaluated on; the non-Play id's 2.5 m spacing changes what the
# cameras see past the desk and would confound cross-checkpoint deltas.
TASK=Rebot-Workstation-PickPlace1-VisionDR-Play-v0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p "$OUT"

# name | extra env vars (RE3SIM_VDR group switches; unlisted groups forced off below)
ROWS=(
  "nominal        RE3SIM_VDR=0"
  "cam_1x         RE3SIM_VDR_CAM=1"
  "cam_1.5x       RE3SIM_VDR_CAM=1 RE3SIM_VDR_SCALE=1.5"
  "focal_1x       RE3SIM_VDR_FOCAL=1"
  "wrist_1x       RE3SIM_VDR_WRIST=1"
  "light_1x       RE3SIM_VDR_LIGHT=1"
  "light_1.5x     RE3SIM_VDR_LIGHT=1 RE3SIM_VDR_SCALE=1.5"
  "bg_1x          RE3SIM_VDR_BG=1"
  "all_1x         RE3SIM_VDR_CAM=1 RE3SIM_VDR_FOCAL=1 RE3SIM_VDR_WRIST=1 RE3SIM_VDR_LIGHT=1 RE3SIM_VDR_BG=1"
  "all_1.5x       RE3SIM_VDR_CAM=1 RE3SIM_VDR_FOCAL=1 RE3SIM_VDR_WRIST=1 RE3SIM_VDR_LIGHT=1 RE3SIM_VDR_BG=1 RE3SIM_VDR_SCALE=1.5"
)

for row in "${ROWS[@]}"; do
  name=$(echo "$row" | awk '{print $1}')
  vars=$(echo "$row" | cut -d' ' -f2- | sed 's/^ *//')
  json="$OUT/$name.json"
  if [[ -f "$json" ]]; then
    # resume only if the existing row is for THIS checkpoint — a stale row from another
    # ckpt reused silently would corrupt the whole matrix
    if python -c "import json,sys; sys.exit(0 if json.load(open('$json')).get('ckpt')=='$CKPT' else 1)" 2>/dev/null; then
      echo "== $name: exists for this ckpt, skipping"
      continue
    fi
    echo "== $name: exists but for a DIFFERENT ckpt — re-running"
  fi
  echo "== $name: $vars"
  # groups default ON inside visual_dr.py, so single-axis rows must force the others OFF;
  # master switch and scale are pinned so nothing leaks in from the caller's environment
  env RE3SIM_SPLATS_PER_ENV=1 \
      RE3SIM_VDR=1 RE3SIM_VDR_SCALE=1.0 \
      RE3SIM_VDR_CAM=0 RE3SIM_VDR_FOCAL=0 RE3SIM_VDR_WRIST=0 \
      RE3SIM_VDR_LIGHT=0 RE3SIM_VDR_BG=0 \
      $vars \
      python -u re3sim/act/eval_flow_vision.py --headless \
        --task "$TASK" --num_envs "$N" --seeds "$SEEDS" \
        --ckpt "$CKPT" --out "$json"
done

python - "$OUT" <<'EOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
print(f"\n{'row':<12} {'success':>8} {'no_lift':>8} {'near_miss':>10} {'dropped':>8}")
for f in sorted(out.glob("*.json")):
    try:
        p = json.load(open(f))["pooled"]
        print(f"{f.stem:<12} {p['success']:>7.1%} {p['no_lift']:>7.1%} "
              f"{p['near_miss']:>9.1%} {p['dropped']:>7.1%}")
    except Exception as ex:  # noqa: BLE001
        print(f"{f.stem:<12} unreadable: {ex}")
EOF
