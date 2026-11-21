from custom_llm.generation.qa_generator import QAPair
from custom_llm.generation.quality import filter_pairs, is_consistent
from custom_llm.knowledge import Fact

FACT = Fact(
    id="reembolso",
    text="Reembolsos são aceitos em até 7 dias corridos.",
    key_terms=("7 dias",),
    doc_title="Reembolsos",
)


def test_answers_missing_the_key_term_are_dropped():
    pairs = [
        QAPair("reembolso", "Qual é o prazo para pedir reembolso?", "O prazo é de 30 dias para qualquer caso."),
    ]
    kept, report = filter_pairs(pairs, {"reembolso": FACT}, dedup_threshold=0.9)

    assert kept == []
    assert report["inconsistent_with_fact"] == 1


def test_near_duplicate_questions_are_removed():
    pairs = [
        QAPair("reembolso", "Qual é o prazo para pedir reembolso?", "O prazo é de 7 dias corridos após a cobrança."),
        QAPair("reembolso", "qual e o prazo para pedir reembolso", "Você pode pedir em até 7 dias corridos."),
    ]
    kept, report = filter_pairs(pairs, {"reembolso": FACT}, dedup_threshold=0.9)

    assert len(kept) == 1
    assert report["near_duplicate"] == 1


def test_consistency_check_uses_key_terms():
    pair = QAPair("reembolso", "Qual é o prazo para reembolso?", "Em até 7 dias corridos após a cobrança.")
    assert is_consistent(pair, FACT)
