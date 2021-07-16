import torch
import numpy as np


def get_pad_mask(seq, pad_idx):
    # create mask with the values of 0's wherever there is padding in the input
    # seq: [b, seq_len]

    pad_mask = (seq != pad_idx)
    pad_mask = pad_mask.unsqueeze(-2)
    return pad_mask


def get_subsequent_mask(seq):
    '''
        [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] x batch_size
    '''
    _, seq_len = seq.size()
    subsequent_mask = (1 - torch.triu(torch.ones((1, seq_len,
                       seq_len), device=seq.device), diagonal=1)).bool()
    return subsequent_mask
