import torch

from models.model import Transformer

if __name__ == '__main__':
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)

    model = Transformer(n_src_vocab=100,
                        n_trg_vocab=100,
                        src_pad_idx=0,
                        trg_pad_idx=0,
                        max_len=5000,
                        d_model=512,
                        d_ffn=2048,
                        n_layer=6,
                        n_head=8,
                        dropout=0.1)
    
    model = model.to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters} trainable parameters.')

    src_seq = torch.randint(50, (1, 5)).to(device)
    trg_seq = torch.randint(50, (1, 5)).to(device)
    pred = model(src_seq, trg_seq)
    print(pred.shape)