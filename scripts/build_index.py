"""Step 3: embed the knowledge base and persist a FAISS index for the RAG approaches.

The index is built from the current knowledge, which includes the post-training updates. That is
the point of RAG: the knowledge can change without retraining the model.
"""

from __future__ import annotations

import argparse
import logging

from custom_llm.config import load_config
from custom_llm.knowledge import load_knowledge_base, load_updates
from custom_llm.retrieval import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)

    base_kb = load_knowledge_base(cfg.paths.knowledge_base)
    update_version, updates = load_updates(cfg.paths.knowledge_updates)
    current_kb = base_kb.with_updates(updates, version=update_version)

    retriever = Retriever(cfg.retrieval.embedding_model, cfg.retrieval.top_k)
    retriever.build(current_kb)
    retriever.save(cfg.paths.index_dir)
    logging.info("Indexed %d passages into %s", len(retriever.passages), cfg.paths.index_dir)


if __name__ == "__main__":
    main()
