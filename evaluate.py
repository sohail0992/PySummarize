"""
Evaluation script: compute loss, perplexity, BLEU, and ROUGE on the test set
"""

import os
import math
import json

# enable MPS fallback for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from models.transformer import Transformer
from utils.dataset import CodeSummarizationDataset, collate_fn, PAD_ID
from utils.inference import load_model, load_tokenizer, greedy_decode, get_device


def evaluate():
    print("Starting evaluation...")

    checkpoint_path = "checkpoints/best_model_v2.pt"
    tokenizer_path = "data/tokenizer.json"
    num_bleu_samples = 500  # greedy decode is slow, so we evaluate BLEU/ROUGE on a subset

    device = get_device()
    print(f"Using device: {device}")

    # load model and tokenizer
    model = load_model(checkpoint_path, device)
    tokenizer = load_tokenizer(tokenizer_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint["vocab_size"]
    print(f"Loaded model from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")

    # load test set
    print("Loading test dataset...")
    test_dataset = CodeSummarizationDataset("test", tokenizer_path)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    print(f"Test examples: {len(test_dataset)}")

    # --- 1. LOSS AND PERPLEXITY ---
    print("\nComputing test loss and perplexity...")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            code_ids = batch["code_ids"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_target = batch["decoder_target"].to(device)

            output = model(code_ids, decoder_input)
            output = output.view(-1, vocab_size)
            decoder_target = decoder_target.view(-1)

            loss = criterion(output, decoder_target)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Perplexity: {perplexity:.2f}")

    # --- 2. BLEU AND ROUGE ---
    print(f"\nComputing BLEU and ROUGE on {num_bleu_samples} examples (greedy decode)...")

    # download nltk data for tokenization if not already present
    nltk.download("punkt_tab", quiet=True)

    references = []
    hypotheses = []
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    examples = []  # store a few qualitative examples

    for i in range(min(num_bleu_samples, len(test_dataset))):
        code = test_dataset.data[i]["code"]
        ref_docstring = test_dataset.data[i]["docstring"]

        # generate prediction using greedy decode
        pred_docstring = greedy_decode(model, tokenizer, code, device=device)

        # BLEU: needs tokenized words as lists
        ref_tokens = nltk.word_tokenize(ref_docstring.lower())
        pred_tokens = nltk.word_tokenize(pred_docstring.lower())
        references.append([ref_tokens])
        hypotheses.append(pred_tokens)

        # ROUGE: works on raw strings
        rouge_result = scorer.score(ref_docstring, pred_docstring)
        rouge1_scores.append(rouge_result["rouge1"].fmeasure)
        rouge2_scores.append(rouge_result["rouge2"].fmeasure)
        rougeL_scores.append(rouge_result["rougeL"].fmeasure)

        # save first 5 as qualitative examples
        if len(examples) < 5:
            examples.append({
                "code": code[:200] + ("..." if len(code) > 200 else ""),
                "reference": ref_docstring,
                "predicted": pred_docstring,
            })

        if (i + 1) % 100 == 0:
            print(f"  Decoded {i+1}/{num_bleu_samples}")

    # compute corpus BLEU with smoothing (handles short predictions)
    smooth = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smooth)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    print(f"\nBLEU Score: {bleu_score:.4f}")
    print(f"ROUGE-1 F1: {avg_rouge1:.4f}")
    print(f"ROUGE-2 F1: {avg_rouge2:.4f}")
    print(f"ROUGE-L F1: {avg_rougeL:.4f}")

    # --- 3. QUALITATIVE EXAMPLES ---
    print("\n" + "=" * 60)
    print("QUALITATIVE EXAMPLES")
    print("=" * 60)
    for idx, ex in enumerate(examples):
        print(f"\n--- Example {idx+1} ---")
        print(f"Code:      {ex['code']}")
        print(f"Reference: {ex['reference']}")
        print(f"Predicted: {ex['predicted']}")

    # --- 4. SAVE RESULTS ---
    results = {
        "test_loss": avg_loss,
        "test_perplexity": perplexity,
        "bleu": bleu_score,
        "rouge1": avg_rouge1,
        "rouge2": avg_rouge2,
        "rougeL": avg_rougeL,
        "num_bleu_samples": num_bleu_samples,
        "examples": examples,
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to evaluation_results.json")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    evaluate()
