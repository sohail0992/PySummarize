"""
Main training script
"""

import json
import os
import math

# enable MPS fallback for unsupported ops (falls back to CPU for those ops)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from models.transformer import Transformer
from utils.dataset import CodeSummarizationDataset, collate_fn, PAD_ID


# Noam learning rate schedule from "Attention Is All You Need"
# ramps up linearly for warmup_steps, then decays with 1/sqrt(step)
def get_lr(step, d_model, warmup_steps):
    if step == 0:
        step = 1
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)


def train():
    print("Starting training...")

    tokenizer_path = "data/tokenizer.json"
    checkpoint_dir = "checkpoints"
    history_path = "training_history.json"

    # config
    num_epochs = 10 # no of iterations over the entire dataset
    batch_size = 32 # batch size is how many examples we process before updating model weights
    d_model = 512 
    warmup_steps = 4000 # number of steps to linearly increase learning rate at the start of training
    grad_clip = 1.0 # gradient clippign like bell curve and find the activaiton zone where the model learns best, prevents exploding gradients

    os.makedirs(checkpoint_dir, exist_ok=True)

    # cuda for nvidia, mps for apple silicon, otherwise cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # we need vocab_size from the tokenizer to initialize the model
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    # load train and validation splits
    print("Loading training dataset...")
    train_dataset = CodeSummarizationDataset("train", tokenizer_path)
    print("Loading validation dataset...")
    val_dataset = CodeSummarizationDataset("validation", tokenizer_path)

    # get ids from batch collate_fn in utils/dataset.py
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    # initialize the model with vocab_size and pad_id for masking, then move to device
    model = Transformer(vocab_size=vocab_size, pad_id=PAD_ID)
    model.to(device)

    # count trainable parameters for reference
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # lr=1.0 because the Noam scheduler controls the actual learning rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: get_lr(step, d_model, warmup_steps)
    )

    # ignore PAD tokens when computing loss
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # track loss per epoch for plotting later
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # --- TRAIN ---
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            code_ids = batch["code_ids"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_target = batch["decoder_target"].to(device)

            # model is transformer.py, it takes code_ids and decoder_input 
            # IDs => Embeddings => Positional Encoding => Transformer Layers.
            output = model(code_ids, decoder_input)

            # cross-entropy expects (N, C) so we flatten batch and seq_len into one dimension
            output = output.view(-1, vocab_size)
            decoder_target = decoder_target.view(-1)

            # compute loss by comparing model output to target summary tokens, ignoring PAD tokens
            loss = criterion(output, decoder_target)

            optimizer.zero_grad()
            loss.backward()
            # clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            num_train_batches += 1

            if (batch_idx + 1) % 500 == 0:
                avg_so_far = total_train_loss / num_train_batches
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg_so_far:.4f} | LR: {current_lr:.2e}")

        avg_train_loss = total_train_loss / num_train_batches

        # --- VALIDATE ---
        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                code_ids = batch["code_ids"].to(device)
                decoder_input = batch["decoder_input"].to(device)
                decoder_target = batch["decoder_target"].to(device)

                output = model(code_ids, decoder_input)
                output = output.view(-1, vocab_size)
                decoder_target = decoder_target.view(-1)

                loss = criterion(output, decoder_target)
                total_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches

        # perplexity = e^loss, lower = better
        train_perplexity = math.exp(avg_train_loss)
        val_perplexity = math.exp(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} (PPL: {train_perplexity:.2f}) | "
              f"Val Loss: {avg_val_loss:.4f} (PPL: {val_perplexity:.2f})")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # save checkpoint if best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "vocab_size": vocab_size,
                "d_model": d_model,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"  Saved best model (val_loss={avg_val_loss:.4f})")

    # save for plotting in notebooks/training.ipynb
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    print("Training complete!")


if __name__ == "__main__":
    train()
