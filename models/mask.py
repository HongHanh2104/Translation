import torch
import numpy as np

def get_pad_mask(seq, pad_idx):
    # create mask with the values of 0's wherever there is padding in the input
    # seq: [b, seq_len]
    
    pad_mask = (seq != pad_idx).unsqueeze(-2)
    return pad_mask
    

def get_subsequent_mask(seq):
    # create mask to prevent the decoder from pointing to 
    # the future position in the sequence
    # seq_len = seq.size(1)
    # np_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1).astype('uint8')
    # np_mask = Variable(torch.from_numpy(np_mask) == 0)
    # return np_mask
    
    batch_size, seq_len = seq.size()
    subsequent_mask = (1 - torch.triu(torch.ones((1, seq_len, seq_len), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

if __name__ == '__main__':
    src = torch.randint(0, 5, (1, 8))
    print(src)
    mask = get_pad_mask(src, 6, 0) & get_subsequent_mask(src)
    print(mask)
