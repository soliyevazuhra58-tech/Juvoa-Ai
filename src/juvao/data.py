"""Dataset validation, normalization, deduplication and chat formatting."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset


def normalize_record(record: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(record.get("messages"), list):
        messages = record["messages"]
    elif "prompt" in record and "response" in record:
        messages = [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]
    else:
        raise ValueError("Expected 'messages' or both 'prompt' and 'response' fields.")

    cleaned: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            continue
        if content:
            cleaned.append({"role": role, "content": content})
    if not cleaned or not any(m["role"] == "assistant" for m in cleaned):
        raise ValueError("Conversation must contain non-empty assistant content.")
    return cleaned


def load_jsonl(path: str | Path) -> list[list[dict[str, str]]]:
    conversations = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                conversation = normalize_record(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid record at {path}:{line_no}: {exc}") from exc
            key = json.dumps(conversation, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                conversations.append(conversation)
    if not conversations:
        raise ValueError(f"No valid conversations found in {path}")
    return conversations


def prepare_splits(input_file: str, output_dir: str, validation_ratio: float = 0.05, seed: int = 42):
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1).")
    conversations = load_jsonl(input_file)
    random.Random(seed).shuffle(conversations)
    n_val = int(len(conversations) * validation_ratio)
    validation = conversations[:n_val]
    train = conversations[n_val:]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({"messages": row}, ensure_ascii=False) + "\n")
    return len(train), len(validation)


def to_text_dataset(path: str, tokenizer) -> Dataset:
    conversations = load_jsonl(path)
    rows = []
    for messages in conversations:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        rows.append({"text": text})
    return Dataset.from_list(rows)
