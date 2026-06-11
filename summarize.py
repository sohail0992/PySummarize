"""
CLI for generating docstrings from Python code using the trained model.

Single snippet:
    python summarize.py --input "def add(x, y): return x + y"

Whole .py file — raw source (default):
    python summarize.py --file path/to/code.py

Whole .py file — strip # comments and docstrings before passing to model:
    python summarize.py --file path/to/code.py --strip-comments

Whole .ipynb notebook — one summary per code cell:
    python summarize.py --file path/to/notebook.ipynb

Add --code to also show code previews alongside summaries.
"""

import os
import ast
import io
import json
import tokenize
import argparse

# enable MPS fallback for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from utils.inference import load_model, load_tokenizer, beam_search_decode, get_device


def _strip(raw: str, node=None) -> str:
    """Remove # comments and existing docstring from raw source, preserving indentation."""
    lines = raw.splitlines()

    # docstring line range (relative to snippet start)
    doc_rows = set()
    if node is not None:
        body = node.body
        if (body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            rel_start = body[0].lineno - node.lineno
            rel_end   = body[0].end_lineno - node.lineno
            doc_rows  = set(range(rel_start, rel_end + 1))

    # # comment column per line
    comment_col = {}
    try:
        for tok_type, _, (srow, scol), _, _ in tokenize.generate_tokens(
                io.StringIO(raw).readline):
            if tok_type == tokenize.COMMENT:
                comment_col[srow - 1] = scol
    except tokenize.TokenError:
        pass

    result = []
    for i, line in enumerate(lines):
        if i in doc_rows:
            continue
        if i in comment_col:
            line = line[:comment_col[i]].rstrip()
        result.append(line)
    return "\n".join(result)


def extract_definitions(source: str, strip_comments: bool = False) -> list[tuple[str, str]]:
    """Return (name, snippet) for every function/class in *source*."""
    tree = ast.parse(source)
    lines = source.splitlines()
    items = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            raw = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            snippet = _strip(raw, node) if strip_comments else raw
            items.append((node.name, snippet))
    return items


def extract_cells(path: str, strip_comments: bool = False) -> list[tuple[str, str]]:
    """Return (label, source) for every non-empty code cell in a notebook."""
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    items = []
    for i, cell in enumerate(nb.get("cells", []), 1):
        if cell.get("cell_type") != "code":
            continue
        raw = "".join(cell.get("source", []))
        if not raw.strip():
            continue
        snippet = _strip(raw) if strip_comments else raw
        items.append((f"Cell {i}", snippet))
    return items


def main():
    parser = argparse.ArgumentParser(description="Generate docstrings from Python code")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str,
                       help='Single code snippet, e.g. "def add(x, y): return x + y"')
    group.add_argument("--file", type=str,
                       help="Python .py or .ipynb file to summarize")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_v2.pt",
                        help="Path to model checkpoint (default: checkpoints/best_model_v2.pt)")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json",
                        help="Path to tokenizer file (default: data/tokenizer.json)")
    parser.add_argument("--code", action="store_true",
                        help="Also print code previews alongside each summary")
    parser.add_argument("--strip-comments", action="store_true",
                        help="Strip # comments and docstrings before passing to the model")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model     = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer)
    print("Model loaded.\n")

    # --- single snippet mode ---
    if args.input:
        summary = beam_search_decode(model, tokenizer, args.input, device=device)
        print(f"Code:    {args.input}")
        print(f"Summary: {summary}")
        return

    # --- file mode: .py or .ipynb ---
    sc = args.strip_comments
    if args.file.endswith(".ipynb"):
        items = extract_cells(args.file, strip_comments=sc)
        label = "cell(s)"
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            source = f.read()
        items = extract_definitions(source, strip_comments=sc)
        label = "definition(s)"

    if not items:
        print(f"Nothing to summarize in '{args.file}'.")
        return

    mode_note = " (comments stripped)" if sc else ""
    print(f"Analyzing '{args.file}' ({len(items)} {label}){mode_note}...\n")
    summaries = []
    for name, snippet in items:
        n_tokens = len(tokenizer.encode(snippet).ids)
        passed   = min(n_tokens, 128)
        truncated = " (TRUNCATED)" if n_tokens > 128 else ""
        print(f"  {name}: {passed}/{n_tokens} tokens passed{truncated}")
        summary = greedy_decode(model, tokenizer, snippet, device=device)
        summaries.append((name, snippet, summary))
    print()

    # --- default: numbered steps overview ---
    if not args.code:
        print(f"What '{args.file}' does:\n")
        for i, (name, _, summary) in enumerate(summaries, 1):
            one_line = " ".join(summary.split())
            print(f"  Step {i} ({name}): {one_line.rstrip('.')}")
    else:
        for name, snippet, summary in summaries:
            one_line = " ".join(summary.split())
            print(f"{name}: {one_line.rstrip('.')}")
            preview = "\n  ".join(snippet.splitlines()[:5])
            if len(snippet.splitlines()) > 5:
                preview += "\n  ..."
            print(f"  Code:\n  {preview}")
            print()


if __name__ == "__main__":
    main()
