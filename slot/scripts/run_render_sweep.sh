#!/usr/bin/env bash
# Render-quality sweep — Big Will: "THE QUALITY OF THE PNG is awful. it is like super noisy."
#
# He is right and it is my setting, not the renderer. G0 ran antialiasing_mode="Off" (chosen to
# dodge EXP08's DLSS suspicion) and Isaac Lab defaults samples_per_pixel = 1. One sample per
# pixel with no AA is grainy and aliased -- and it renders the SAME grainy image every time,
# which is why G0's static-render probe read 0.0 and I wrongly reported that as "clean". It
# proved temporal determinism, not spatial quality.
#
# Every variant pins the identical robot/block pose (champion driven 150 steps with fixed_x0
# zeros), so the PNGs are directly comparable. `ss4_spp8` is rendered LAST as the reference
# other variants are scored against -- but it is also the leading candidate fix:
#
#   supersampling (render 640x360, box-average to 160x90) gives 16 samples per training pixel
#   with NO temporal accumulation and NO DLSS -- so it cannot reintroduce the GPU-load
#   dependence that has EXP08's Gate D blocked. It costs render time, nothing else.
#
# Big Will compares <out>/<tag>/{wrist,workspace}_native.png -- all at the training resolution
# the policy actually consumes, which is the only comparison that decides anything.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

OUT="${1:-runs/vision_render_probe}"
probe() { python scripts/vision_render_probe.py --out "$OUT" "$@" 2>&1 | grep -E "^\[|cameras\]" || true; }

# the reference first, so later variants can be scored against it
probe --tag ss4_spp8   --aa Off  --spp 8 --scale 4 --denoiser
REF="$OUT/ss4_spp8"

probe --tag off_spp1   --aa Off  --spp 1               --ref "$REF"   # what G0 shipped
probe --tag off_spp8   --aa Off  --spp 8               --ref "$REF"
probe --tag fxaa_spp8  --aa FXAA --spp 8               --ref "$REF"
probe --tag dlss_spp1  --aa DLSS --spp 1               --ref "$REF"   # Isaac's default
probe --tag ss2_spp8   --aa Off  --spp 8 --scale 2     --ref "$REF"
probe --tag ss4_dn     --aa Off  --spp 4 --scale 4 --denoiser --ref "$REF"

echo
echo "compare these (all 160x90, the resolution the policy consumes):"
ls -1 "$OUT"/*/wrist_native.png
echo "=== render sweep done  $(date -Is) ==="
