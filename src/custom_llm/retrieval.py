"""Dense retrieval over fact passages: sentence embeddings indexed with FAISS."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from custom_llm.knowledge import KnowledgeBase


@dataclass(frozen=True)
class Passage:
    fact_id: str
    text: str


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


class Retriever:
    """E5-family encoders expect "passage: " and "query: " prefixes, applied here, not by callers."""

    def __init__(self, encoder_name: str, top_k: int) -> None:
        self.encoder = SentenceTransformer(encoder_name)
        self.top_k = top_k
        self.passages: list[Passage] = []
        self.index: faiss.Index | None = None

    def build(self, kb: KnowledgeBase) -> None:
        self.passages = [Passage(fact.id, fact.passage) for fact in kb.facts]
        vectors = self._encode([f"passage: {p.text}" for p in self.passages])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str) -> list[Hit]:
        if self.index is None:
            raise RuntimeError("The index is not built or loaded.")
        vector = self._encode([f"query: {query}"])
        scores, ids = self.index.search(vector, self.top_k)
        return [
            Hit(self.passages[i], float(score))
            for score, i in zip(scores[0], ids[0])
            if i != -1
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss.index"))
        payload = [asdict(p) for p in self.passages]
        (directory / "passages.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, encoder_name: str, top_k: int) -> Retriever:
        retriever = cls(encoder_name, top_k)
        retriever.index = faiss.read_index(str(directory / "faiss.index"))
        raw = json.loads((directory / "passages.json").read_text(encoding="utf-8"))
        retriever.passages = [Passage(**item) for item in raw]
        return retriever

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype="float32")
