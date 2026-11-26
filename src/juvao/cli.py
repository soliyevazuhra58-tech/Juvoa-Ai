"""Command-line interface."""
import argparse

from .data import prepare_splits
from .evaluate import evaluate
from .inference import JuvaoChat
from .train import train


def main():
    parser = argparse.ArgumentParser(prog="juvao")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-data")
    prep.add_argument("--input", required=True)
    prep.add_argument("--output", default="data/processed")
    prep.add_argument("--validation-ratio", type=float, default=0.05)

    fit = sub.add_parser("train")
    fit.add_argument("--config", default="configs/train.yaml")

    chat = sub.add_parser("chat")
    chat.add_argument("--adapter", default=None)
    chat.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")

    ev = sub.add_parser("evaluate")
    ev.add_argument("--adapter", required=True)
    ev.add_argument("--prompts", default="data/eval/prompts.jsonl")
    ev.add_argument("--output", default="outputs/evaluation.jsonl")

    args = parser.parse_args()
    if args.command == "prepare-data":
        train_n, val_n = prepare_splits(args.input, args.output, args.validation_ratio)
        print(f"Prepared {train_n} training and {val_n} validation conversations.")
    elif args.command == "train":
        train(args.config)
    elif args.command == "chat":
        bot = JuvaoChat(args.adapter, args.base_model)
        history = []
        print("Juvao AI — type /exit to quit, /reset to clear history.")
        while True:
            prompt = input("You: ").strip()
            if prompt == "/exit":
                break
            if prompt == "/reset":
                history = []
                continue
            if not prompt:
                continue
            history.append({"role": "user", "content": prompt})
            answer = bot.reply(history)
            print(f"Juvao: {answer}")
            history.append({"role": "assistant", "content": answer})
    elif args.command == "evaluate":
        evaluate(args.adapter, args.prompts, args.output)


if __name__ == "__main__":
    main()
