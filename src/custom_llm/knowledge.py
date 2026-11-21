"""Structured knowledge base: documents made of atomic, checkable facts."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Fact:
    """An atomic statement about the product and the terms a correct answer must contain."""

    id: str
    text: str
    key_terms: tuple[str, ...]
    doc_id: str = ""
    doc_title: str = ""

    @property
    def passage(self) -> str:
        return f"{self.doc_title}: {self.text}" if self.doc_title else self.text


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    facts: tuple[Fact, ...]


@dataclass(frozen=True)
class FactUpdate:
    """A change made after training. The probe is written by hand to test stale knowledge."""

    fact_id: str
    new_text: str
    key_terms: tuple[str, ...]
    probe_question: str
    probe_answer: str


@dataclass(frozen=True)
class KnowledgeBase:
    product: str
    version: str
    documents: tuple[Document, ...]

    @property
    def facts(self) -> list[Fact]:
        return [fact for doc in self.documents for fact in doc.facts]

    def fact_by_id(self) -> dict[str, Fact]:
        return {fact.id: fact for fact in self.facts}

    def with_updates(self, updates: list[FactUpdate], version: str) -> KnowledgeBase:
        known = self.fact_by_id()
        unknown = sorted(u.fact_id for u in updates if u.fact_id not in known)
        if unknown:
            raise ValueError(f"Updates reference unknown facts: {unknown}")

        changed = {u.fact_id: u for u in updates}
        documents = tuple(
            Document(doc.id, doc.title, tuple(_apply(fact, changed) for fact in doc.facts))
            for doc in self.documents
        )
        return KnowledgeBase(self.product, version, documents)


def _apply(fact: Fact, changed: dict[str, FactUpdate]) -> Fact:
    update = changed.get(fact.id)
    if update is None:
        return fact
    return replace(fact, text=update.new_text, key_terms=update.key_terms)


def load_knowledge_base(path: Path) -> KnowledgeBase:
    raw = json.loads(path.read_text(encoding="utf-8"))
    documents = tuple(
        Document(
            id=doc["id"],
            title=doc["title"],
            facts=tuple(
                Fact(
                    id=item["id"],
                    text=item["text"],
                    key_terms=tuple(item["key_terms"]),
                    doc_id=doc["id"],
                    doc_title=doc["title"],
                )
                for item in doc["facts"]
            ),
        )
        for doc in raw["documents"]
    )
    kb = KnowledgeBase(product=raw["product"], version=raw["version"], documents=documents)

    ids = [fact.id for fact in kb.facts]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate fact ids in the knowledge base.")
    return kb


def load_updates(path: Path) -> tuple[str, list[FactUpdate]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    updates = [
        FactUpdate(
            fact_id=change["fact_id"],
            new_text=change["new_text"],
            key_terms=tuple(change["key_terms"]),
            probe_question=change["probe_question"],
            probe_answer=change["probe_answer"],
        )
        for change in raw["changes"]
    ]
    return raw["version"], updates
