"""Small paired qualitative comparison of base model and Juvao adapter."""
from __future__ import annotations

import json
from pathlib import Path

from .inference import JuvaoChat


def evaluate(adapter: str, prompts_file: str, output_file: str = "outputs/evaluation.jsonl"):
    base = JuvaoChat(adapter=None)
    tuned = JuvaoChat(adapter=adapter)
    rows = []
    with Path(prompts_file).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = str(item["prompt"])
            messages = [{"role": "user", "content": prompt}]
            row = {
                "prompt": prompt,
                "base": base.reply(messages),
                "juvao": tuned.reply(messages),
            }
            rows.append(row)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_file).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Compared {len(rows)} prompts. Results: {output_file}")
