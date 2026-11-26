"""Load the base model and optional PEFT adapter for chat."""
from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class JuvaoChat:
    def __init__(self, adapter: str | None = None, base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"torch_dtype": "auto"} if self.device == "cuda" else {}
        self.model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs).to(self.device)
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    @torch.inference_mode()
    def reply(self, messages: list[dict[str, str]], max_new_tokens: int = 180,
              temperature: float = 0.7, top_p: float = 0.9) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
