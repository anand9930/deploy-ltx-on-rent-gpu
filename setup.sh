#!/bin/bash
# BentoML setup script — runs during Docker image build.
# Clones the official LTX-2 repo and installs ltx-core / ltx-pipelines.
set -e

git lfs install
git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2
pip install --no-cache-dir \
    -e /app/LTX-2/packages/ltx-core \
    -e /app/LTX-2/packages/ltx-pipelines
