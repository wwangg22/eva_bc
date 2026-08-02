#!/bin/bash
# Re-sync curated files from the source project into this repo's working tree.
# READ-ONLY with respect to the source tree; overwrites tracked copies here.
# Deliberately does NOT commit — review the diff, then commit yourself.
set -eu

SRC="/home/william/Desktop/isaacLab/reBot/reBot_ACT"
DST="$(cd "$(dirname "$0")" && pwd)"

# Size cap: never bring in anything over 5 MB.
copy() {  # copy <dest_dir> <files...>
    local dest="$1"; shift
    mkdir -p "$dest"
    local f
    for f in "$@"; do
        [ -f "$f" ] || continue
        if [ "$(stat -c%s "$f")" -gt 5242880 ]; then
            echo "SKIP (>5MB): $f"
            continue
        fi
        cp "$f" "$dest/"
    done
}

# act/: code, configs, vendored-ACT license + provenance.
copy "$DST/act" "$SRC"/act/*.py "$SRC"/act/*.yaml "$SRC"/act/LICENSE "$SRC"/act/PROVENANCE.md

# expert/: code, cuRobo robot configs, batch scripts.
# Excluded by omission: *.h5 *.pt *.log *.json *.mp4 *.png (data/artifacts/run outputs).
copy "$DST/expert" "$SRC"/expert/*.py "$SRC"/expert/*.yml "$SRC"/expert/*.sh

# experiments/: runnable code + chain scripts (docs go to docs/experiments below).
# Excluded by omission: *.json results, *.pt probe weights.
copy "$DST/experiments" "$SRC"/experiments/*.py "$SRC"/experiments/*.sh

# docs/ re-layout: top-level .md -> docs/, experiments .md -> docs/experiments/,
# expert write-ups -> docs/expert/.
copy "$DST/docs" "$SRC"/*.md
copy "$DST/docs/experiments" "$SRC"/experiments/*.md
copy "$DST/docs/expert" "$SRC"/expert/lying_tz_probe_results.md "$SRC"/expert/place_fail_analysis_raw.md

echo
echo "Sync complete. Nothing was committed: review \`git diff\`, then commit."
