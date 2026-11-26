"""Optional PEFT adapter merge utility; merged weights are larger than the adapter."""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="outputs/juvao-merged")
    args = parser.parse_args()

    kwargs = {"torch_dtype": "auto"}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    base = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)
    tuned = PeftModel.from_pretrained(base, args.adapter)
    merged = tuned.merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.save_pretrained(args.output)
    print(f"Merged model exported to {args.output}")


if __name__ == "__main__":
    main()
