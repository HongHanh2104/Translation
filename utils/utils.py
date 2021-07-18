import os
import logging
import torch

PAD_INDEX = 0

def idx_to_word(x, vocab):
    # words = []
    # for i in x:
    #     word = vocab.itos[i]
    #     if '<' not in word:
    #         words.append(word)
    # words = " ".join(words)

    words = []
    key_list = list(vocab.keys())
    value_list = list(vocab.values())

    for item in x:
        word = key_list[value_list.index(item)]
        if '<' not in word:
            words.append(word)
    words = " ".join(words)
    return words
   
def input_target_collate_fn(batch):
    """merges a list of samples to form a mini-batch."""

    # indexed_sources = [sources for sources, targets in batch]
    # indexed_targets = [targets for sources, targets in batch]

    sources_lengths = [len(sources) for sources, targets in batch]
    targets_lengths = [len(targets) for sources, targets in batch]

    sources_max_length = max(sources_lengths)
    targets_max_length = max(targets_lengths)

    sources_padded = [sources + [PAD_INDEX] * (sources_max_length - len(sources)) for sources, targets in batch]
    targets_padded = [targets + [PAD_INDEX] * (targets_max_length - len(targets)) for sources, targets in batch]

    sources_tensor = torch.tensor(sources_padded)
    targets_tensor = torch.tensor(targets_padded)

    return sources_tensor, targets_tensor
