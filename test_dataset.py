from dataset.multi30k import Multi30kLoader_tmp, Multi30kLoader
from utils.tokenize import Tokenizer

from torch.utils.data import DataLoader
from torchtext.legacy.data import Field, BucketIterator

if __name__ == '__main__':
    data = Multi30kLoader(ext=('.en', '.de'))

    train, valid, _ = data.create_dataset()
    
    data.build_vocab(data=train, min_freq=2)
    
    train_iter, val_iter, _ = data.make_iter(batch_size=1, 
                                             device='cpu')
    
    src_pad_idx, trg_pad_idx = data.get_pad_idx()

    enc_voc_size, dec_voc_size = data.get_voc_size()

    print(enc_voc_size, dec_voc_size)

    for i, b in enumerate(train_iter):
        print("src: ", b.src)
        print("*"*50)
        print("trg: ", b.trg) 
        break                                   
