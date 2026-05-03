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

Best model checkpoint saves to `checkpoints/best_model.pt` and training history to `training_history.json`.

### 5. Evaluate the model

Computes test loss, perplexity, BLEU, and ROUGE scores on the test set (14,918 examples). BLEU/ROUGE are computed on a 500-example subset using greedy decoding. Results are saved to `evaluation_results.json`.

```bash
python evaluate.py
```

### 6. Generate summaries

**Single snippet:**
```bash
python summarize.py --input "def add(x, y): return x + y"
```

**Whole Python file** — summarizes every function and class, one step per definition:
```bash
python summarize.py --file utils/inference.py
```

**Whole Python file with comment/docstring stripping** — strips `#` comments and existing docstrings before passing to the model:
```bash
python summarize.py --file utils/inference.py --strip-comments
```

**Jupyter notebook** — summarizes each code cell:
```bash
python summarize.py --file notebooks/evaluation.ipynb
```

**Show code previews** alongside summaries (works with all modes):
```bash
python summarize.py --file utils/inference.py --code
```

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | — | Single code snippet to summarize |
| `--file` | — | `.py` or `.ipynb` file to summarize |
| `--strip-comments` | off | Strip `#` comments and docstrings before inference |
| `--code` | off | Print code previews alongside summaries |
| `--checkpoint` | `checkpoints/best_model.pt` | Path to model checkpoint |
| `--tokenizer` | `data/tokenizer.json` | Path to tokenizer file |
