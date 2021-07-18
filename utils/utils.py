import torch
import numpy as np

PAD_INDEX = 0
END_INDEX = 3

def idx_to_word(x, vocab):
    #     word = vocab.itos[i]

    # words = []
    # key_list = list(vocab.keys())
    # value_list = list(vocab.values())

    # for item in x:
    #     word = key_list[value_list.index(item)]
    #     if '<' not in word:
    #         words.append(word)
    # words = " ".join(words)
    words = []
    for item in x:
        word = vocab[item]
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

def preprocess(trg, out):
    trg_end_pos = np.where(trg == END_INDEX)[0][0]
    trg = trg[:trg_end_pos + 1]
    out_end_pos = np.where(out == END_INDEX)
    if len(out_end_pos[0]) > 0:
        out_end_pos = out_end_pos[0][0]
        out = out[:out_end_pos + 1]
    return trg, out

if __name__ == '__main__':
    trg = np.array([5, 9, 8, 7, 7, 3, 0, 0, 0, 0])
    out = np.array([6, 8, 7, 9, 4, 4, 3, 200, 200, 200, 200])
    result = preprocess(trg, out)

    print(result[0])
    print(result[1])