import torch
from torch import nn

class TokenCrossEntropyLoss(nn.Module):
    def __init__(self, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.base_cross_entropy = nn.CrossEntropyLoss(reduction='sum',
                                                      ignore_index=pad_idx)
        
    def forward(self, out, trg):
        #print("outs: ", outs.size())
        #print("trgs: ", trgs.size())
        ntokens = (trg != self.pad_idx).sum().item()

        out = out.reshape(-1, out.shape[-1])
        trg = trg.reshape(-1)

        loss = self.base_cross_entropy(out, trg)      

        return loss, ntokens



    