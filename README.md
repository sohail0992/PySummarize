# ML-Based Python Code Summarization

A PyTorch encoder-decoder Transformer trained from scratch to generate natural language docstrings from Python source code.

## Requirements

- Python 3.10+

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

### 6. Generate summaries

**Example 1 — Parse JSON file (7 lines):**
```bash
python summarize.py --input "def parse_json_file(filepath):
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data"
```

Output: `return json`

**Example 2 — Validate email address (8 lines):**
```bash
python summarize.py --input "def is_valid_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not email:
        return False
    return re.match(pattern, email) is not None"
```

**Example 3 — Calculate statistics (9 lines):**
```bash
python summarize.py --input "def calculate_statistics(numbers):
    if not numbers:
        return None
    total = sum(numbers)
    avg = total / len(numbers)
    minimum = min(numbers)
    maximum = max(numbers)
    return {'average': avg, 'min': minimum, 'max': maximum}"
```

**File summarization (works best):**
```bash
python summarize.py --file utils/inference.py
```

**With code preview:**
```bash
python summarize.py --file utils/inference.py --code
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

**Note:** The model performs best on functions with 7+ lines of code. Shorter functions may produce less coherent summaries due to limited context for the encoder.
