# ML-Based Python Code Summarization

A PyTorch encoder-decoder Transformer trained from scratch to generate natural language docstrings from Python source code.

## Requirements

- Python 3.10+

## Quick Examples

The model achieves best results on functions with 7+ lines (trained with 75% complex functions per batch using class-balanced batch sampling). Here are three examples with self-explanatory function names:

**Example 1 — Find duplicates in list:**
```bash
python summarize.py --input "def find_duplicates(items):
    seen = set()
    duplicates = []
    for item in items:
        if item in seen:
            if item not in duplicates:
                duplicates.append(item)
        else:
            seen.add(item)
    return duplicates"
```
Summary: `Find items in collection`

**Example 2 — Read CSV file:**
```bash
python summarize.py --input "def read_csv_file(filepath):
    import csv
    data = []
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data"
```
Summary: `Read CSV file`

**Example 3 — Filter values by threshold:**
```bash
python summarize.py --input "def filter_by_threshold(values, threshold):
    if not values:
        return []
    filtered = []
    for value in values:
        if value >= threshold:
            filtered.append(value)
    return filtered"
```
Summary: `Parse filter values`

## Model Checkpoints

Two models are included:
- **`best_model_v2.pt`** (default) — Trained with class-balanced batch sampling (25% trivial, 75% complex functions per batch). **Recommended.**
- **`best_model.pt`** — Trained with standard sampling (original 91% complex, 0.95% trivial distribution).

## How to Run

### 1. Create virtual environment

```bash
python3 -m venv mlsa_env
source mlsa_env/bin/activate
```

### 2. Install dependencies

```bash
pip install --timeout 300 -r requirements.txt
```

### 3. Train the tokenizer

Open and run all cells in `data/tokenizer_training.ipynb` in Jupyter. This trains a BPE tokenizer on the CodeXGLUE dataset and saves `data/tokenizer.json`, which is required by all subsequent steps.

```bash
jupyter notebook data/tokenizer_training.ipynb
```

### 4. Train the model

```bash
python train.py
```

Best model checkpoint saves to `checkpoints/best_model_v2.pt` and training history to `logs/training_history.json`.

### 5. Evaluate the model

Computes test loss, perplexity, BLEU, and ROUGE scores on the test set (14,918 examples). BLEU/ROUGE are computed on a 500-example subset using beam search. Results are saved to `logs/evaluation_results.json`.

```bash
python evaluate.py
```

### 6. More options

**File summarization (recommended):**
```bash
python summarize.py --file utils/inference.py
```

**With code preview:**
```bash
python summarize.py --file utils/inference.py --code
```

**Use a different model checkpoint:**
```bash
python summarize.py --input "def read_csv_file(filepath):
    import csv
    data = []
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data" --checkpoint checkpoints/best_model.pt
```

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | — | Single code snippet to summarize |
| `--file` | — | `.py` file to summarize |
| `--strip-comments` | off | Strip `#` comments and docstrings before inference |
| `--code` | off | Print code previews alongside summaries |
| `--checkpoint` | `checkpoints/best_model_v2.pt` | Path to model checkpoint |
| `--tokenizer` | `data/tokenizer.json` | Path to tokenizer file |

## Medium

Read the detailed write-up on Medium: [A Step-by-Step Guide to Building a Python Code Summarizer](https://medium.com/@msohail.se/a-step-by-step-guide-to-building-a-python-code-summarizer-13c10df86b1f)

## References

- **Dataset**: [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) — Microsoft
