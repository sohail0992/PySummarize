"""
CLI for generating docstrings from Python code using the trained model
"""

import os
import argparse

# enable MPS fallback for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from utils.inference import load_model, load_tokenizer, greedy_decode, get_device


def main():
    parser = argparse.ArgumentParser(description="Generate a docstring from Python code")
    parser.add_argument("--input", type=str, required=True,
                        help='Python code to summarize, e.g. "def add(x, y): return x + y"')
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json",
                        help="Path to tokenizer file")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # load model and tokenizer
    model = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer)
    print("Model loaded.\n")

    # generate summary
    summary = greedy_decode(model, tokenizer, args.input, device=device)

    print(f"Code:    {args.input}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
