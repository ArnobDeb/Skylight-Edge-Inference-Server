#!/usr/bin/env python3
"""Download public OpenVINO text-generation models into the sibling models directory.

This project keeps the heavy model assets outside the GitHub repo so the repository stays
lightweight. The actual model directory lives beside the repo at:
    ../models
and the repo contains a small downloader script only.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Only text-generation models are included here.
# Image/video models such as sd-xl-ov and other non-text generation assets are intentionally excluded.
TEXT_GENERATION_MODELS = {
    "qwen3-4b-fp16-ov": "OpenVINO/qwen3-4b-fp16-ov",
    "qwen3-8b-fp16-ov": "OpenVINO/qwen3-8b-fp16-ov",
    "qwen3-14b-fp16-ov": "OpenVINO/qwen3-14b-fp16-ov",
    "gemma-3-4b-it-fp16-ov": "OpenVINO/gemma-3-4b-it-fp16-ov",
    "gemma-3-12b-it-fp16-ov": "OpenVINO/gemma-3-12b-it-fp16-ov",
    "phi-3.5-mini-int4": "OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov",
}

for local_name, repo_id in TEXT_GENERATION_MODELS.items():
    local_path = MODEL_DIR / local_name
    if local_path.exists():
        print(f"[skip] {local_name} already exists at {local_path}")
        continue

    print(f"[pull] {repo_id} -> {local_name}")
    snapshot_download(repo_id=repo_id, local_dir=str(local_path))

print(f"\nDone. Models are stored at: {MODEL_DIR}")
