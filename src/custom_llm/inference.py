"""Answer generation for each approach under comparison."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel

from custom_llm.models import load_tokenizer_and_base_model
from custom_llm.prompts import build_messages
from custom_llm.retrieval import Retriever


@dataclass(frozen=True)
class Answer:
    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    retrieved_fact_ids: tuple[str, ...]


class LocalModel:
    """The base model with a LoRA adapter. The adapter can be switched off to get base behaviour."""

    def __init__(
        self,
        base_model: str,
        adapter_dir: Path,
        max_new_tokens: int,
        load_in_4bit: bool,
    ) -> None:
        self.tokenizer, base = load_tokenizer_and_base_model(base_model, load_in_4bit)
        self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def generate(self, messages: list[dict[str, str]], use_adapter: bool) -> tuple[str, int, int]:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        switch = nullcontext() if use_adapter else self.model.disable_adapter()
        with switch:
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_len = encoded["input_ids"].shape[1]
        new_tokens = output[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text, int(prompt_len), int(new_tokens.shape[0])


@dataclass
class Approach:
    name: str
    model: LocalModel
    retriever: Retriever | None
    use_adapter: bool

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()

        contexts: list[str] = []
        fact_ids: tuple[str, ...] = ()
        if self.retriever is not None:
            hits = self.retriever.search(question)
            contexts = [hit.passage.text for hit in hits]
            fact_ids = tuple(hit.passage.fact_id for hit in hits)

        messages = build_messages(question, contexts)
        text, prompt_tokens, completion_tokens = self.model.generate(messages, self.use_adapter)
        return Answer(
            text=text,
            latency_s=time.perf_counter() - start,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retrieved_fact_ids=fact_ids,
        )


def build_approaches(model: LocalModel, retriever: Retriever) -> dict[str, Approach]:
    """Four variants on one model: no adaptation (base), RAG only, fine-tuning only, and both."""
    return {
        "base": Approach("base", model, None, use_adapter=False),
        "rag": Approach("rag", model, retriever, use_adapter=False),
        "finetuned": Approach("finetuned", model, None, use_adapter=True),
        "hybrid": Approach("hybrid", model, retriever, use_adapter=True),
    }
