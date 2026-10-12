"""Publish a trained Haruspex model folder to the Hugging Face Hub.

Writes a model card (README.md) into the folder, then uploads everything
(the merged GLiNER2.5-Decide model + tokenizer + calibrator.json + card).

  pip install -e ".[data]"          # brings huggingface_hub
  huggingface-cli login             # or set HF_TOKEN
  python scripts/publish_hf.py --model runs/haruspex --repo <you>/haruspex
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haruspex.card import write_card  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Publish Haruspex to the HF Hub")
    ap.add_argument("--model", required=True, help="trained model dir")
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. TyrusRC/haruspex")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    args = ap.parse_args()

    model_dir = Path(args.model)
    # Accept either the adapter artifact (haruspex.json + a LoRA adapter) or a
    # standalone merged model dir (config.json).
    is_adapter = (model_dir / "haruspex.json").exists()
    is_merged = (model_dir / "config.json").exists()
    if not (is_adapter or is_merged):
        sys.exit(f"error: {model_dir} is not a Haruspex model dir (no haruspex.json or config.json)")

    card = write_card(model_dir, args.repo)
    print(f"[publish] wrote model card -> {card}  ({'adapter' if is_adapter else 'merged'})")

    from huggingface_hub import HfApi
    api = HfApi(token=args.token or None)
    api.create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")
    print(f"[publish] uploading {model_dir} -> https://huggingface.co/{args.repo}")
    # Ship the artifact, not the training bookkeeping (logs, failed-batch dumps, configs).
    api.upload_folder(folder_path=str(model_dir), repo_id=args.repo, repo_type="model",
                      ignore_patterns=["logs/*", "training_config.json", "failed_batches.jsonl",
                                       "**/optimizer.pt", "**/scheduler.pt"])
    print(f"[publish] done -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
