#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "tenacity>=9.0.0",
#     "httpx>=0.28.1",
#     "instructor>=1.7.0",
#     "google-genai>=1.68.0",
#     "pydantic>=2.12.5",
#     "sqlite-vec>=0.1.3",
#     "scikit-learn>=1.5.0",
#     "scikit-learn-intelex",
#     "umap-learn>=0.5.11",
#     "numpy>=1.24.0",
#     "markitdown>=0.0.1a4",
#     "feedparser>=6.0.12",
#     "lxml>=5.1.0",
#     "llama-cpp-python>=0.2.56",
#     "jsonref>=1.1.0",
# ]
# ///

# -*- coding: utf-8 -*-
# @description RAPTOR Memory Consolidation (v1.0) - SOTA Wrapper.

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

from vromlix_utils import VromlixRaptorEngine, vromlix

# --- SOTA SILENCE ---
warnings.filterwarnings("ignore", category=UserWarning, module="umap")
logging.getLogger("sklearnex").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# --- SOTA Intel Optimization ---
try:
    from sklearnex import patch_sklearn

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with Path(os.devnull).open("w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            patch_sklearn()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
except ImportError:
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VROMLIX RAPTOR Consolidation Engine")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Delete current hierarchy and perform global consolidation.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="vromlix_memory.sqlite",
        help="Nombre de la base de datos a consolidar (ej. media_transcripts.sqlite)",
    )
    args = parser.parse_args()

    # Centralized Engine Initialization
    db_target = str(vromlix.paths.databases / args.db)
    print(f"🦅 Launching RAPTOR SOTA Consolidator on {args.db}...")
    engine = VromlixRaptorEngine(db_path=db_target)
    try:
        engine.run_consolidation(force_full=args.full)
    except KeyboardInterrupt:
        print("\n Process interrupted by user.")
    except Exception as e:
        print(f"\n Critical Error: {e}")
