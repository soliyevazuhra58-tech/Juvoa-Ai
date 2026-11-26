# Juvao AI

Juvao AI is a compact English conversational model built by fine-tuning **Qwen2.5-0.5B-Instruct** with LoRA/QLoRA. It reuses the original Qwen tokenizer and pretrained weights; it is not trained from scratch.

## Project status

This repository contains the training and inference pipeline. A trained checkpoint is **not included**. Run training and evaluation in your own Kaggle session before describing any adapter or model as trained.

## Features

- Qwen2.5-0.5B-Instruct base model
- Original Qwen tokenizer (no custom BPE)
- PyTorch, Transformers, Datasets, PEFT and TRL-compatible workflow
- LoRA by default; optional 4-bit QLoRA on supported CUDA GPUs
- JSONL conversation dataset validation, cleaning, deduplication and splitting
- Checkpointing, resume support and configurable training
- Base-vs-fine-tuned evaluation and interactive inference
- Adapter export in standard PEFT format; optional merged model export
- Automatic CPU/GPU selection in CLI and Kaggle notebook

## Quick start

Python 3.10+ is recommended. Training and inference automatically use CUDA when available, otherwise CPU. A CUDA GPU is strongly recommended for practical fine-tuning speed.

```bash
git clone https://github.com/soliyevazuhra58-tech/Juvoa-Ai.git
cd Juvoa-Ai
pip install -e .
python -m juvao.cli prepare-data --input data/raw/conversations.jsonl --output data/processed
python -m juvao.cli train --config configs/train.yaml
python -m juvao.cli chat --adapter outputs/juvao-lora
```

For a Kaggle-first workflow, open `notebooks/Juvao_AI_Kaggle_Fine_Tuning.ipynb` and run cells in order. Set the dataset and output paths in the first configuration cell.

## Dataset format

One JSON object per line. Supported formats:

```json
{"messages":[{"role":"user","content":"Hello!"},{"role":"assistant","content":"Hi! How can I help?"}]}
```

or

```json
{"prompt":"Hello!","response":"Hi! How can I help?"}
```

Use only data you have permission to use. Remove private, sensitive, copyrighted, or otherwise unauthorized content. The included sample is illustrative only.

## Configuration

Edit `configs/train.yaml` to set the base model, dataset paths, LoRA settings, training hyperparameters, precision, and output directory. Defaults are intentionally conservative for a 0.5B model. GPU memory and runtime vary; reduce sequence length, batch size, or enable gradient accumulation if needed.

## Training and outputs

Training writes checkpoints and the final adapter under `outputs/juvao-lora` by default. The adapter is not a standalone base model: inference loads the base Qwen model and then attaches the adapter. Merging is optional and requires enough memory.

## Evaluation

```bash
python -m juvao.cli evaluate --adapter outputs/juvao-lora --prompts data/eval/prompts.jsonl
```

The evaluation script compares base and adapter generations on the same prompts. This is a simple qualitative comparison, not a benchmark or proof of improved capability.

## Automatic device selection

The training and inference code automatically selects CUDA when a supported GPU is available; otherwise it uses CPU. 4-bit QLoRA is enabled only on CUDA with bitsandbytes. CPU mode uses full-precision weights and standard LoRA (no 4-bit quantization). CPU fine-tuning can be substantially slower and may require reducing batch size and sequence length.

## Reproducibility

Record the exact dataset version, base model revision, package versions, random seed, and training configuration for each run. Do not upload private datasets, access tokens, or large model weights to Git.

## License

See [LICENSE](LICENSE). The Qwen model has its own license and usage terms; review the upstream model card before use: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct.
