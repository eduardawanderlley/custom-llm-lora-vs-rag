import pytest

from custom_llm.dataset_builder import FactSplit, build_dataset, check_leakage, split_facts
from custom_llm.generation.qa_generator import QAPair
from custom_llm.knowledge import Document, Fact, FactUpdate, KnowledgeBase


def make_kb(n: int = 5) -> KnowledgeBase:
    facts = tuple(
        Fact(id=f"f{i}", text=f"Regra {i} vale v{i}.", key_terms=(f"v{i}",), doc_title="Regras")
        for i in range(n)
    )
    return KnowledgeBase("Fluxo", "1.0", (Document("regras", "Regras", facts),))


def make_pairs(n: int = 5, per_fact: int = 2) -> list[QAPair]:
    return [
        QAPair(f"f{i}", f"Pergunta {i} sobre a regra {j}?", f"Resposta {i} com v{i}.")
        for i in range(n)
        for j in range(per_fact)
    ]


SPLIT = FactSplit(train_fact_ids=("f0", "f1", "f2"), heldout_fact_ids=("f3", "f4"))


def test_split_is_disjoint_complete_and_deterministic():
    ids = [f"f{i}" for i in range(10)]
    split = split_facts(ids, heldout_ratio=0.3, seed=7)

    assert set(split.train_fact_ids).isdisjoint(split.heldout_fact_ids)
    assert len(split.train_fact_ids) + len(split.heldout_fact_ids) == 10
    assert split == split_facts(ids, heldout_ratio=0.3, seed=7)


def test_heldout_facts_never_reach_training_data():
    rows, items = build_dataset(make_pairs(), make_kb(), updates=[], split=SPLIT, seed=0)

    trained_text = " ".join(row["completion"][0]["content"] for row in rows)
    assert "v3" not in trained_text
    assert "v4" not in trained_text

    assert sum(item.regime == "seen" for item in items) == 3
    assert sum(item.regime == "unseen" for item in items) == 4


def test_leakage_check_flags_shared_questions():
    kb = make_kb()
    rows, items = build_dataset(make_pairs(), kb, updates=[], split=SPLIT, seed=0)
    check_leakage(rows, items)

    leaked_row = {
        "prompt": [{"role": "system", "content": "x"}, {"role": "user", "content": items[0].question}],
        "completion": [{"role": "assistant", "content": "y"}],
    }
    with pytest.raises(ValueError):
        check_leakage(rows + [leaked_row], items)


def test_updated_facts_are_probed_and_excluded_from_seen():
    kb = make_kb()
    update = FactUpdate(
        fact_id="f0",
        new_text="Regra 0 vale w0.",
        key_terms=("w0",),
        probe_question="Qual o valor da regra 0?",
        probe_answer="A regra 0 vale w0.",
    )
    current = kb.with_updates([update], version="1.1")
    _, items = build_dataset(make_pairs(), current, [update], SPLIT, seed=0)

    assert any(item.regime == "update" and item.fact_id == "f0" for item in items)
    assert not any(item.regime == "seen" and item.fact_id == "f0" for item in items)
    assert current.fact_by_id()["f0"].text == "Regra 0 vale w0."


def test_updates_for_unknown_facts_are_rejected():
    update = FactUpdate("inexistente", "x", ("x",), "q?", "a")
    with pytest.raises(ValueError):
        make_kb().with_updates([update], version="1.1")
