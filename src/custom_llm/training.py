"""LoRA and QLoRA fine-tuning with TRL's SFTTrainer.

Training uses the prompt/completion format, so the loss is computed on the answer tokens only.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from custom_llm.config import TrainingConfig
from custom_llm.models import load_tokenizer_and_base_model

logger = logging.getLogger(__name__)


def train(cfg: TrainingConfig, train_file: Path, output_dir: Path, seed: int) -> dict:
    tokenizer, model = load_tokenizer_and_base_model(cfg.base_model, cfg.load_in_4bit)
    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    dataset = load_dataset("json", data_files=str(train_file), split="train")
    logger.info("Training on %d examples", len(dataset))

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=cfg.max_seq_length,
        logging_steps=5,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )

    start = time.perf_counter()
    result = trainer.train()
    runtime = time.perf_counter() - start

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    summary = {
        "train_examples": len(dataset),
        "trainable_parameters": trainable,
        "train_runtime_s": round(runtime, 1),
        "final_train_loss": result.training_loss,
        "loss_curve": [
            {"step": entry["step"], "loss": entry["loss"]}
            for entry in trainer.state.log_history
            if "loss" in entry
        ],
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
