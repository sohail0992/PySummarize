"""
Data loading and processing utilities
"""

import torch
from torch.utils.data import Dataset
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
