"""CLI entry point to download models before starting the API server."""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from download_models import ensure_models_downloaded

model_dir = os.path.join(
    os.getenv("VOLUME_MOUNT_PATH", "/runpod-volume"), "models"
)
ensure_models_downloaded(model_dir)
