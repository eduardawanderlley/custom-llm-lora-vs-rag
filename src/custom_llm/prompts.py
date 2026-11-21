"""Prompt construction shared by dataset generation, fine-tuned inference and RAG."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "Você é o assistente de suporte do Fluxo, um sistema de faturamento e emissão de "
    "notas fiscais. Responda em português, de forma objetiva."
)

RAG_INSTRUCTIONS = (
    "Use somente as informações do contexto. Se o contexto não trouxer a resposta, diga "
    "que não encontrou essa informação, sem complementar com conhecimento externo."
)


def build_messages(question: str, contexts: list[str] | None = None) -> list[dict[str, str]]:
    """Return chat messages. Without contexts, the prompt matches the training format exactly."""
    if not contexts:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

    numbered = "\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT} {RAG_INSTRUCTIONS}"},
        {"role": "user", "content": f"Contexto:\n{numbered}\n\nPergunta: {question}"},
    ]
