"""LoRA/optional QLoRA fine-tuning entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .data import to_text_dataset


def train(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model_cfg, data_cfg, lora_cfg, train_cfg = (
        cfg["model"], cfg["data"], cfg["lora"], cfg["training"]
    )
    use_cuda = torch.cuda.is_available()
    use_4bit = bool(train_cfg.get("use_4bit", False) and use_cuda)
    use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported() and train_cfg.get("bf16", True))
    compute_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"Training device: {device_name}; 4-bit QLoRA: {use_4bit}")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": model_cfg.get("trust_remote_code", False)}
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        except ImportError as exc:
            raise RuntimeError("Install bitsandbytes to use 4-bit QLoRA.") from exc
    else:
        load_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"], **load_kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = to_text_dataset(data_cfg["train_file"], tokenizer)
    eval_path = data_cfg.get("validation_file")
    eval_ds = to_text_dataset(eval_path, tokenizer) if eval_path and Path(eval_path).exists() else None
    max_len = int(model_cfg.get("max_seq_length", 1024))

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    if eval_ds is not None:
        eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text"])

    use_fp16 = bool(use_cuda and not use_bf16)

    args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(train_cfg["eval_steps"]) if eval_ds is not None else None,
        save_total_limit=int(train_cfg["save_total_limit"]),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        seed=int(train_cfg.get("seed", 42)),
        optim="adamw_torch",
        logging_dir=str(Path(train_cfg["output_dir"]) / "logs"),
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"))
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    print(f"Adapter and tokenizer saved to {train_cfg['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    train(parser.parse_args().config)
