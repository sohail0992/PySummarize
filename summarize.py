"""
CLI for generating docstrings from Python code using the trained model.

Single snippet:
    python summarize.py --input "def add(x, y): return x + y"

Whole file — summaries only (default):
    python summarize.py --file path/to/code.py

Whole file — with code previews:
    python summarize.py --file path/to/code.py --code
"""

import os
import ast
import argparse

# enable MPS fallback for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from utils.inference import load_model, load_tokenizer, greedy_decode, get_device


def extract_definitions(source: str) -> list[tuple[str, str]]:
    """Return (name, code_snippet) for every function/class in *source*."""
    tree = ast.parse(source)
    lines = source.splitlines()
    items = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            snippet = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            items.append((node.name, snippet))
    return items


def main():
    parser = argparse.ArgumentParser(description="Generate docstrings from Python code")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str,
                       help='Single code snippet, e.g. "def add(x, y): return x + y"')
    group.add_argument("--file", type=str,
                       help="Python source file — summarizes every function/class in it")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json",
                        help="Path to tokenizer file")
    parser.add_argument("--code", action="store_true",
                        help="Also print code previews alongside each summary")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer)
    print("Model loaded.\n")

    # --- single snippet mode ---
    if args.input:
        summary = greedy_decode(model, tokenizer, args.input, device=device)
        print(f"Code:    {args.input}")
        print(f"Summary: {summary}")
        return

    # --- whole-file mode ---
    with open(args.file, "r", encoding="utf-8") as f:
        source = f.read()

    definitions = extract_definitions(source)
    if not definitions:
        print("No function or class definitions found.")
        return

    print(f"Analyzing '{args.file}' ({len(definitions)} definition(s))...\n")
    summaries = []
    for name, snippet in definitions:
        summary = greedy_decode(model, tokenizer, snippet, device=device)
        summaries.append((name, snippet, summary))

    # --- default: numbered steps overview ---
    if not args.code:
        print(f"What '{args.file}' does:\n")
        for i, (name, _, summary) in enumerate(summaries, 1):
            one_line = " ".join(summary.split())
            print(f"  Step {i} ({name}): {one_line.rstrip('.')}")
    else:
        for name, snippet, summary in summaries:
            print(f"{name}: {summary}")
            preview = "\n  ".join(snippet.splitlines()[:5])
            if len(snippet.splitlines()) > 5:
                preview += "\n  ..."
            print(f"  Code:\n  {preview}")
            print()


if __name__ == "__main__":
    main()
