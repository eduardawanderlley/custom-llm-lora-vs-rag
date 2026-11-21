"""Shared helpers: reproducibility, JSON I/O, parsing and text normalization."""

from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_json(raw: str) -> dict[str, Any] | None:
    """Parse the first JSON object in a model response, tolerating surrounding prose."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def normalize_text(text: str) -> str:
    """Lowercase, strip accents and punctuation, and collapse whitespace.

    Every text comparison in the project goes through this function, so a term matches
    regardless of case, accents or decimal separators.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    unaccented = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^\w\s]", " ", unaccented)
    return re.sub(r"\s+", " ", cleaned).strip()


def term_present(text: str, term: str) -> bool:
    """True if the term appears as whole tokens in the text.

    A term may list alternatives separated by "|", for example "e-mail|email". Any one of them
    is enough, which keeps a correct answer from being penalised for a spelling variant.
    """
    haystack = f" {normalize_text(text)} "
    return any(f" {normalize_text(alt)} " in haystack for alt in term.split("|"))
