#!/usr/bin/env bash
# demos -> flow-matching BC -> batched sim eval, end to end.
#
# Split from the expert smoke test on purpose: the smoke run is a decision point, not a stage.
# If the expert's success rate is not worth training on, more batches of it are not either --
# so the gate is read by a human (or by the caller of this script) before this runs.
#
# Every stage writes under re3sim/runs/<tag>/ so two variants -- primitive colliders vs
# reconstructed meshes -- never overwrite each other's numbers. Which one an env used is
# printed by the env itself at load time.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
cd /home/eva/Desktop/isaacLab/eva_bc

TAG=${TAG:-recon}
NENV=${NENV:-128}
BATCHES=${BATCHES:-9}
STEPS=${STEPS:-60000}
SEED=${SEED:-1}
RUN=re3sim/runs/$TAG
mkdir -p "$RUN"

echo "############ [1/3] demonstrations  $(date -Is) ############"
python -u re3sim/expert/collect_demos.py --headless \
  --num_envs "$NENV" --batches "$BATCHES" --seed "$SEED" \
  --out "$RUN/demos.hdf5" 2>&1 | tee "$RUN/collect.log" | grep -aE "batch|SUCCESS|taxonomy|demos|Error|Traceback"

echo
echo "############ [2/3] flow-matching BC  $(date -Is) ############"
python -u re3sim/act/train_flow.py --data "$RUN/demos.hdf5" --out "$RUN/bc" \
  --steps "$STEPS" --seed "$SEED" 2>&1 | tee "$RUN/train.log" | tail -20

echo
echo "############ [3/3] evaluation  $(date -Is) ############"
# Held-out spawn seeds. `collect_demos` uses seed*1000 + batch, so seed 1 occupies
# 1000..1008 -- 88000+ cannot collide with it. A policy scored on the layouts it trained on
# measures memorisation.
python -u re3sim/act/eval_flow.py --headless --ckpt "$RUN/bc/ckpt_final.pt" \
  --num_envs 128 --seeds 88000,88001,88002 \
  --out "$RUN/bc_eval.json" 2>&1 | tee "$RUN/eval.log" | tail -40

echo
echo "############ done  $(date -Is) ############"
