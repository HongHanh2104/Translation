import os
import logging
import torch

PAD_INDEX = 0

def idx_to_word(x, vocab):
    words = []
    for i in x:
        word = vocab.itos[i]
        if '<' not in word:
            words.append(word)
    words = " ".join(words)
    return words
   
def input_target_collate_fn(batch):
    """merges a list of samples to form a mini-batch."""

    # indexed_sources = [sources for sources, inputs, targets in batch]
    # indexed_inputs = [inputs for sources, inputs, targets in batch]
    # indexed_targets = [targets for sources, inputs, targets in batch]

    sources_lengths = [len(sources) for sources, inputs, targets in batch]
    inputs_lengths = [len(inputs) for sources, inputs, targets in batch]
    targets_lengths = [len(targets) for sources, inputs, targets in batch]

    sources_max_length = max(sources_lengths)
    inputs_max_length = max(inputs_lengths)
    targets_max_length = max(targets_lengths)

    sources_padded = [sources + [PAD_INDEX] * (sources_max_length - len(sources)) for sources, inputs, targets in batch]
    inputs_padded = [inputs + [PAD_INDEX] * (inputs_max_length - len(inputs)) for sources, inputs, targets in batch]
    targets_padded = [targets + [PAD_INDEX] * (targets_max_length - len(targets)) for sources, inputs, targets in batch]

    sources_tensor = torch.tensor(sources_padded)
    inputs_tensor = torch.tensor(inputs_padded)
    targets_tensor = torch.tensor(targets_padded)

    return sources_tensor, inputs_tensor, targets_tensor
