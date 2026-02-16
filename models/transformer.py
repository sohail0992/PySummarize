"""
Transformer model architecture
"""

import math
import torch
import torch.nn as nn


# Summary:
# The Transformer maps each input token to a 512-dimension embedding for both the source
# and target sequences. These sequences include special SOS (Start of Sentence) and EOS
# (End of Sentence) tokens to manage sequence boundaries. In the forward method, we create
# padding masks so both the encoder and decoder layers ignore Pad IDs in both sequences.
# Through Positional Encoding, we inject sequence order into these embeddings using sine
# and cosine functions of varying frequencies.


# To the Transformer, for x in range(5) looks exactly the same as range(5) x in for.
# Without position, the model can't tell them apart.

# max_len is maximum length for GPU memory, the GPU create a table with 5000 rows safe upper limit
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 5k position * 512 dimensions = 2.5 million values,
        # this is the table that the model will look up to get positional information
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    # If you pass 10 IDs, you get 10 Vectors back. Those vectors have been:
    # Looked up in the dictionary.
    # Multiplied by a scaling factor.
    # Added to a position value.
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x




# The Transformer model consists of three main components:
# 1. The Lookup Table (Embedding): This layer converts token IDs from the dataset into
#    dense vector representations. Each token ID is mapped to a vector of size d_model.
# 2. The Transformer Module: This is the core of the architecture, which processes the
#    input code and generates the output docstring. It consists of multiple layers of
#    self-attention and feedforward networks for both the encoder and decoder.
# 3. The Output Layer: This final linear layer projects the output of the transformer to
#    the size of the vocabulary, allowing us to predict the next token in the docstring
#    by outputting a probability distribution over the vocabulary for each position in the output sequence.

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nheads=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, pad_id=1):
        super(Transformer, self).__init__()
        # we get the d_model which is the size of the vectors in the transformer
        self.d_model = d_model
        self.pad_id = pad_id

        # 1. THE LOOKUP TABLE (Embedding)
        # Embedding Layer: (IDs from dataset getitem => Vector)
            # config embedding layer to convert token ids to dense vectors of size d_model
            # while specifying the padding index
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # 2 POSITIONAL ENCODING
        # inject position info into embeddings so the model knows token order
        self.positional_encoding = PositionalEncoding(d_model)

        # 3. THE TRANSFORMER MODULE
        # Transformer (Encoder/Decoder): (Vector => Smarter Vector)
        # imagine if for x in range(6) 
        # smarter_vector will attach extra important info that x is the iterater
        # using the neighborhood of x it understands that x is being used in a loop
        # and is not just a random variable
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        # 4. THE OUTPUT LAYER
        # Linear/Output Layer: (Smarter Vector =>  Vocab Scores)
        #  y = xW + b
        self.fc_out = nn.Linear(d_model, vocab_size)

    # code_ids = code tokens (encoder input), decoder_input = docstring tokens (decoder input)
    def forward(self, code_ids, decoder_input):
        # how many tokens in sequence
        target_len = decoder_input.size(1)
        # create a mask to prevent the model from looking at future tokens in the target sequence during training
        target_mask = self.transformer.generate_square_subsequent_mask(target_len).to(decoder_input.device)

        # to not consider the PAD 1 which is just placeholder to match sequence lengths
        src_key_padding_mask = (code_ids == self.pad_id)
        target_key_padding_mask = (decoder_input == self.pad_id)

        # embed the input and target sequences
        # scale embeddings by sqrt(d_model) as in "Attention Is All You Need"
        # then add positional encoding so the model knows token order
        code_embedded = self.positional_encoding(self.embedding(code_ids) * math.sqrt(self.d_model))
        decoder_embedded = self.positional_encoding(self.embedding(decoder_input) * math.sqrt(self.d_model))

        # pass through the transformer
        transformer_output = self.transformer(
            src=code_embedded,
            tgt=decoder_embedded,
            tgt_mask=target_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=target_key_padding_mask,
        )
        # project the transformer output to the vocabulary size
        output = self.fc_out(transformer_output)
        # (batch_size, seq_len, vocab_size)
        return output
