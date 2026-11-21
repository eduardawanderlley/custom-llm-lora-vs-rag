"""Tokenizer and base model loading, shared by training and inference."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_tokenizer_and_base_model(base_model: str, load_in_4bit: bool = False):
    cuda = torch.cuda.is_available()
    quantization = None
    if load_in_4bit and cuda:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16 if cuda else torch.float32,
        device_map="auto" if cuda else None,
    )
    return tokenizer, model
