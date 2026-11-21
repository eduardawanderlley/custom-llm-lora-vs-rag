from custom_llm.evaluation.metrics import exact_match, is_refusal, key_term_recall, token_f1


def test_exact_match_ignores_case_accents_and_punctuation():
    assert exact_match("O plano é Starter.", "o plano e starter") == 1.0


def test_token_f1_gives_partial_credit():
    score = token_f1("o plano custa 89 reais", "o plano custa 89 reais por mês")
    assert 0.0 < score < 1.0


def test_key_term_recall_matches_whole_numbers_only():
    assert key_term_recall("O valor é 189 reais", ("89",)) == 0.0
    assert key_term_recall("O valor é 89 reais", ("89",)) == 1.0


def test_key_term_recall_handles_decimal_comma():
    assert key_term_recall("Cada nota extra custa R$ 0,40.", ("0,40",)) == 1.0


def test_key_term_recall_is_partial_across_terms():
    assert key_term_recall("Até 14 dias, sem cartão", ("14 dias", "50")) == 0.5


def test_refusal_detection():
    assert is_refusal("Não encontrei essa informação na base.")
    assert not is_refusal("O plano Starter custa 89 reais por mês.")


def test_key_term_alternatives_count_as_one_requirement():
    assert key_term_recall("Atendimento por email", ("e-mail|email",)) == 1.0
    assert key_term_recall("Atendimento por telefone", ("e-mail|email",)) == 0.0
