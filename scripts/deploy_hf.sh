#!/bin/sh
# Upload the working tree to the Hugging Face Space as a snapshot.
#
# The Space's git server refuses any binary file in the pushed history that
# is not in LFS, and the course-project commits contain PDFs and PNGs, so
# pushing the branch itself does not work. `hf upload` sends the files
# without history and stores binaries the way HF wants.
#
#   pip install huggingface_hub   (once)
#   hf auth login                  (once, paste a write token)
#   scripts/deploy_hf.sh
set -e
cd "$(dirname "$0")/.."

SPACE="${HF_SPACE:-jeffjeff20120112/sheet-to-audio}"
HF=hf
[ -x .venv/bin/hf ] && HF=.venv/bin/hf

# everything the Dockerfile does not need stays out; the sample pages under
# web/public are needed, so only the course-project images are excluded
"$HF" upload "$SPACE" . . --repo-type space --delete "*" \
  --exclude ".git/*" --exclude ".venv/*" --exclude "data/*" --exclude "results/*" \
  --exclude "docs/*" --exclude "tests/*" --exclude "web/node_modules/*" --exclude "web/dist/*" \
  --exclude "models/*.keras" --exclude "*.pdf" --exclude "*.ipynb" --exclude "poster.png" \
  --exclude ".playwright-mcp/*" --exclude "**/__pycache__/*" --exclude "*.egg-info/*" \
  --exclude ".pytest_cache/*" --exclude ".DS_Store" \
  --commit-message "Deploy $(git rev-parse --short HEAD)"

echo "uploaded; build log: https://huggingface.co/spaces/$SPACE?logs=build"
