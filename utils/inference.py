"""
Inference utilities: load a trained model and generate summaries via greedy decoding
"""

import os
import math
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from models.transformer import Transformer
from utils.dataset import PAD_ID, SOS_ID, EOS_ID

# enable MPS fallback for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def get_device():
    """Pick best available device: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device=None):
    """
    Load a trained Transformer from a checkpoint file.

    Returns (model, tokenizer) both ready for inference.
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint["vocab_size"]
    d_model = checkpoint["d_model"]

    model = Transformer(vocab_size=vocab_size, d_model=d_model, pad_id=PAD_ID)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def load_tokenizer(tokenizer_path="data/tokenizer.json"):
    """Load the BPE tokenizer and attach the ByteLevel decoder for readable output."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def greedy_decode(model, tokenizer, code_text, max_len=128, device=None):
    """
    Generate a docstring using beam search (default beam_width=5).
    Falls back to greedy decoding when beam_width=1.

    Beam search keeps track of the top N most likely sequences at each step,
    allowing the model to find better overall sentences instead of greedily
    committing to one token at a time.
    """
    return beam_search_decode(model, tokenizer, code_text, max_len=max_len,
                              beam_width=5, device=device)


def beam_search_decode(model, tokenizer, code_text, max_len=128, beam_width=5, device=None):
    """
    Generate a docstring using beam search.

    Each beam is a tuple of (token_ids_list, cumulative_log_probability).
    At each step we expand all beams, keep the top beam_width candidates,
    and stop when the best beam ends with [EOS].
    """
    if device is None:
        device = next(model.parameters()).device

    # tokenize the source code
    code_ids = tokenizer.encode(code_text).ids[:max_len]
    code_tensor = torch.tensor([code_ids], dtype=torch.long, device=device)

    # each beam: (list of token ids, cumulative log probability)
    beams = [([SOS_ID], 0.0)]
    completed = []

    with torch.no_grad():
        for _ in range(max_len):
            all_candidates = []

            for seq, score in beams:
                # if this beam already ended, keep it as-is
                if seq[-1] == EOS_ID:
                    completed.append((seq, score))
                    continue

                decoder_tensor = torch.tensor([seq], dtype=torch.long, device=device)
                output = model(code_tensor, decoder_tensor)

                # get log probabilities for the last position
                log_probs = torch.log_softmax(output[0, -1, :], dim=-1)

                # take top beam_width tokens
                top_log_probs, top_indices = log_probs.topk(beam_width)

                for i in range(beam_width):
                    token = top_indices[i].item()
                    new_score = score + top_log_probs[i].item()
                    new_seq = seq + [token]
                    all_candidates.append((new_seq, new_score))

            if not all_candidates:
                break

            # keep top beam_width candidates (highest log prob = least negative)
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

            # if best beam ended with EOS, we're done
            if beams[0][0][-1] == EOS_ID:
                completed.append(beams[0])
                break

    # pick the best completed sequence, or best beam if none completed
    if completed:
        completed.sort(key=lambda x: x[1], reverse=True)
        best_seq = completed[0][0]
    else:
        best_seq = beams[0][0]

    # remove [SOS] and [EOS] before decoding
    generated_ids = [t for t in best_seq if t not in (SOS_ID, EOS_ID)]
    summary = tokenizer.decode(generated_ids)
    return summary
