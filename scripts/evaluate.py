"""Step 4: evaluate the four approaches on the test set and write the reports."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from custom_llm.config import load_config
from custom_llm.dataset_builder import load_eval_items
from custom_llm.evaluation.judge import Judge
from custom_llm.evaluation.runner import (
    judge_agreement,
    judge_review_sample,
    run_evaluation,
    summarize,
    summarize_out_of_scope,
)
from custom_llm.generation.llm_client import ChatClient
from custom_llm.inference import LocalModel, build_approaches
from custom_llm.retrieval import Retriever
from custom_llm.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    results_dir = cfg.paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    items = load_eval_items(cfg.paths.test_file)
    logging.info("Evaluation set: %d items", len(items))

    model = LocalModel(
        cfg.training.base_model,
        cfg.paths.adapter_dir,
        cfg.evaluation.max_new_tokens,
        cfg.training.load_in_4bit,
    )
    retriever = Retriever.load(cfg.paths.index_dir, cfg.retrieval.embedding_model, cfg.retrieval.top_k)
    approaches = build_approaches(model, retriever)
    judge = Judge(ChatClient(cfg.evaluation.judge_model, temperature=0.0, max_tokens=300))

    records = run_evaluation(approaches, items, judge)
    records.to_csv(results_dir / "records.csv", index=False)

    by_regime = summarize(records)
    out_of_scope = summarize_out_of_scope(records)
    by_regime.to_csv(results_dir / "summary_by_regime.csv")
    out_of_scope.to_csv(results_dir / "summary_out_of_scope.csv")

    # The review sample is created once. Regenerating it would overwrite the human labels.
    sample_path = results_dir / "judge_review_sample.csv"
    if sample_path.exists():
        agreement = judge_agreement(pd.read_csv(sample_path))
        logging.info("Judge agreement with human labels: %s", agreement or "not labelled yet")
    else:
        judge_review_sample(records, n=40, seed=cfg.seed).to_csv(sample_path, index=False)
        logging.info("Wrote %s for manual labelling", sample_path)

    print(by_regime.to_string())
    print()
    print(out_of_scope.to_string())


if __name__ == "__main__":
    main()
