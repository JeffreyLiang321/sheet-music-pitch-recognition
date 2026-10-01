#!/bin/sh
# Upload the committed tree to the Hugging Face Space as a snapshot.
#
# The Space's git server refuses any binary file in the pushed history that
# is not in LFS, and the course-project commits contain PDFs and PNGs, so
# pushing the branch itself does not work. `hf upload` sends the files
# without history and stores binaries the way HF wants.
#
# Spaces read their configuration (sdk, port) from a YAML block at the top
# of README.md. That block is added here, to the copy that gets uploaded, so
# the README in this repo stays a plain README.
#
#   pip install huggingface_hub   (once)
#   hf auth login                  (once, paste a write token)
#   scripts/deploy_hf.sh           (uploads HEAD, so commit first)
set -e
cd "$(dirname "$0")/.."

SPACE="${HF_SPACE:-jeffjeff20120112/sheet-to-audio}"
HF=hf
[ -x .venv/bin/hf ] && HF=.venv/bin/hf

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
git archive HEAD | tar -x -C "$stage"
{
  printf '%s\n' '---' 'title: Sheet to Audio' 'emoji: 🎼' 'sdk: docker' 'app_port: 7860' 'pinned: false' '---' ''
  git show HEAD:README.md
} > "$stage/README.md"

# everything the Dockerfile does not need stays out; the sample pages under
# web/public are needed, so only the course-project images are excluded
"$HF" upload "$SPACE" "$stage" . --repo-type space --delete "*" \
  --exclude "docs/*" --exclude "tests/*" --exclude "results/*" \
  --exclude "models/*.keras" --exclude "*.pdf" --exclude "*.ipynb" --exclude "poster.png" \
  --commit-message "Deploy $(git rev-parse --short HEAD)"

echo "uploaded; build log: https://huggingface.co/spaces/$SPACE?logs=build"
