"""
Data loading and processing utilities
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tokenizers import Tokenizer
from datasets import load_dataset


# same order as tokenizer_training.ipynb, if we change order here it will mess up the model
UNK_ID = 0
PAD_ID = 1
SOS_ID = 2
EOS_ID = 3


class CodeSummarizationDataset(Dataset):
    def __init__(self, split, tokenizer_path, max_length=128):
        # load the tokenizer we trained and saved in tokenizer_training.ipynb
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length

        # load CodeXGLUE python dataset, split can be "train", "validation", or "test"
        dataset = load_dataset("google/code_x_glue_ct_code_to_text", "python")
        self.data = dataset[split]

    # __ is called magic method
    # when you do len(dataset) it calls this __len__ method and returns 
    # the number of examples in the dataset
    def __len__(self):
        return len(self.data)

    #  when you write ataset[5] it calls this __getitem__ method with idx=5,
    #  and it should return the code and docstring for that example
    def __getitem__(self, idx):
        # grab the code and its docstring from the original dataset,
        #  which is a dictionary with keys "code" and "docstring"
        code = self.data[idx]["code"]
        docstring = self.data[idx]["docstring"]

        # using the tokenizer, convert code and docstring from text to list of token ids
        # we also truncate to max_length to avoid very long sequences that can cause memory issues
        code_ids = self.tokenizer.encode(code).ids[:self.max_length]
        # -1 because we need room for SOS or EOS token
        doc_ids = self.tokenizer.encode(docstring).ids[:self.max_length - 1]

        # we created input with SOS and encoded docstring 
        # target with encoded docstring and EOS token
        decoder_input = [SOS_ID] + doc_ids
        decoder_target = doc_ids + [EOS_ID]

        # torch tensors are the data structure used for model input and output in PyTorch,
        # similar to numpy arrays but with additional functionality for GPU acceleration
        # if you print the decoder_input and decoder_target 
        # they will match except the first and the last which are SOS and EOS
        return {
            "code_ids": torch.tensor(code_ids, dtype=torch.long),
            "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
            "decoder_target": torch.tensor(decoder_target, dtype=torch.long),
        }

# if model input is a batch of 3 sequences with different lengths,
#  after padding (whcih is essentilay placeholder) 
# they will all have the same length, for example:
# [
# [10,  20,  30,  1,  1],  # Function 1 (Padded with two 1s)
# [50,  60,  70,  80, 90],  # Function 2 (No padding needed)
#  [100, 110, 1,   1,  1]   # Function 3 (Padded with three 1s)
# ]
def collate_fn(batch):
    # sequences in a batch can have different lengths, pad shorter ones with PAD token
    # so they all become same size (needed for matrix operations)
    code_ids = [item["code_ids"] for item in batch]
    decoder_inputs = [item["decoder_input"] for item in batch]
    decoder_targets = [item["decoder_target"] for item in batch]

    code_ids = torch.nn.utils.rnn.pad_sequence(code_ids, batch_first=True, padding_value=PAD_ID)
    decoder_inputs = torch.nn.utils.rnn.pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_ID)
    decoder_targets = torch.nn.utils.rnn.pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_ID)

    return {
        "code_ids": code_ids,
        "decoder_input": decoder_inputs,
        "decoder_target": decoder_targets,
    }


class StratifiedBatchSampler(Sampler):
    """
    Create batches with balanced class distribution.
    Instead of random shuffle (which gives 91% complex), ensure each batch has:
    - target_ratio% simple examples (1-3 lines)
    - (1-target_ratio)% complex examples (4+ lines)

    This forces the model to learn both simple and complex patterns equally.
    """

    def __init__(self, dataset, batch_size, target_ratio=0.25):
        """
        Initialize the stratified sampler.

        Args:
            dataset: CodeSummarizationDataset instance with data[idx]["code"]
            batch_size: how many examples per batch (e.g., 32)
            target_ratio: fraction of trivial examples in each batch (e.g., 0.25 = 25%)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_ratio = target_ratio

        # STEP 1: Scan all training examples and separate by complexity
        # We store INDICES (positions), not the actual code
        self.simple_indices = []  # positions of 1-3 line functions
        self.complex_indices = []  # positions of 4+ line functions

        print("Analyzing dataset complexity distribution...")
        for idx in range(len(dataset)):
            # Get code from dataset
            code = dataset.data[idx]["code"]
            # Count lines in this code
            num_lines = len(code.strip().split('\n'))

            # Categorize: simple = 1-3 lines, complex = 4+ lines
            if num_lines <= 3:
                self.simple_indices.append(idx)
            elif num_lines >= 4:
                self.complex_indices.append(idx)

        # Print statistics
        print(f"  Simple (1-3 lines):   {len(self.simple_indices):,}")
        print(f"  Complex (4+ lines):   {len(self.complex_indices):,}")

        # STEP 2: Calculate how many trivial/complex per batch
        # If target_ratio=0.25 and batch_size=32:
        # - num_trivial_per_batch = 32 * 0.25 = 8
        # - num_complex_per_batch = 32 - 8 = 24
        self.num_simple_per_batch = int(batch_size * target_ratio)
        self.num_complex_per_batch = batch_size - self.num_simple_per_batch

        print(f"  Per batch: {self.num_simple_per_batch} simple + {self.num_complex_per_batch} complex")

    def __iter__(self):
        """
        Generate batches with balanced distribution.
        This is called by DataLoader to get batches of indices.

        Yields:
            List of batch_size indices representing one batch
            Each batch has num_trivial_per_batch trivial + num_complex_per_batch complex
        """

        # STEP 1: Make copies and shuffle both lists
        # We shuffle to randomize which examples appear in which batch
        simple_shuffled = self.simple_indices.copy()  # [47, 152, 289, ...]
        complex_shuffled = self.complex_indices.copy()  # [1, 3, 5, 7, ...]
        np.random.shuffle(simple_shuffled)  # Randomize order
        np.random.shuffle(complex_shuffled)

        # STEP 2: Track position in each list as we build batches
        simple_ptr = 0   # Points to next simple to pick
        complex_ptr = 0   # Points to next complex to pick

        # STEP 3: Build batches until we run out of complex examples
        while complex_ptr < len(complex_shuffled):
            batch = []

            # Add simple examples to batch
            for _ in range(self.num_simple_per_batch):
                # Check if we've used all simple examples
                if simple_ptr >= len(simple_shuffled):
                    # Loop back: reset pointer and reshuffle
                    simple_ptr = 0
                    np.random.shuffle(simple_shuffled)

                # Add this simple example's index to batch
                batch.append(simple_shuffled[simple_ptr])
                simple_ptr += 1

            # Add complex examples to batch
            for _ in range(self.num_complex_per_batch):
                # Check if we've run out of complex examples
                if complex_ptr >= len(complex_shuffled):
                    break

                # Add this complex example's index to batch
                batch.append(complex_shuffled[complex_ptr])
                complex_ptr += 1

            # Only yield if batch is full (has exactly batch_size examples)
            if len(batch) == self.batch_size:
                # Shuffle within batch so trivial/complex aren't grouped together
                # This way the model doesn't see patterns like "first 8 are trivial"
                np.random.shuffle(batch)

                # Yield this batch of indices
                # DataLoader will use these to fetch actual examples
                yield batch

    def __len__(self):
        """
        Return number of batches we'll generate per epoch.
        Used by DataLoader for progress bars and epoch logic.

        Returns:
            Approximate number of batches (based on complex examples available)
        """
        # Number of batches = how many complete batches we can make
        return len(self.complex_indices) // self.num_complex_per_batch
