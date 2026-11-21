"""Step 2: LoRA or QLoRA fine-tuning on the synthetic training set."""

from __future__ import annotations

import argparse
import logging

from custom_llm.config import load_config
from custom_llm.training import train
from custom_llm.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    if not cfg.paths.train_file.exists():
        raise SystemExit(f"Missing {cfg.paths.train_file}. Run scripts/generate_dataset.py first.")

    summary = train(cfg.training, cfg.paths.train_file, cfg.paths.adapter_dir, cfg.seed)
    logging.info(
        "Training finished in %.1f s, final loss %.4f",
        summary["train_runtime_s"],
        summary["final_train_loss"],
    )


if __name__ == "__main__":
    main()
