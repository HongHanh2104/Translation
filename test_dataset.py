from dataset.multi30k import Multi30kLoader
from dataset.en_vi_dataset import EN_VIDataset
from dictionaries import IndexDictionary
from torchtext.legacy.data import BucketIterator
from utils.utils import input_target_collate_fn, idx_to_word

from torch.utils.data import DataLoader

if __name__ == '__main__':
    #data = Multi30kLoader(ext=('.en', '.de'))
    data_dir = './data/en-vi/processed-data'

    print('Building dictionaries ...')
    src_dict = IndexDictionary.load(
                    data_dir=data_dir,
                    mode='source')
    print(f'Source vocab size: {src_dict.get_vocab_size()}')

    #print(src_dict.token_index_dict)

    trg_dict = IndexDictionary.load(
                    data_dir=data_dir,
                    mode='target')
    print(f'Target vocab size: {trg_dict.get_vocab_size()}')
    
    train_data = EN_VIDataset(
                    data_dir= data_dir,
                    phase='train')
    
    val_data = EN_VIDataset(
                    data_dir= data_dir,
                    phase='val')

    train_iter, valid_iter = BucketIterator.splits(
                                (train_data, val_data),
                                batch_size=8,
                                device='cuda',
                                shuffle=True,
                                sort=False,
                                sort_within_batch=True,
                                sort_key=lambda x: len(x['text']),
                            )
    # trg = val_data[0][1] 
    # trg = trg[1:]

    # words = idx_to_word(x=trg, vocab=trg_dict.get_vocab_dict())
    # print(words)

    
    # train_dataloader = DataLoader(
    #                 train_data,
    #                 batch_size=8,
    #                 shuffle=True,
    #                 collate_fn=input_target_collate_fn
    #                 )

   
    for i, (srcs, trgs) in enumerate(train_iter):
        srcs.to('cuda')
        trgs.to('cuda')
        break
    #       break
    #       #print("src: ", b.src)
    #     print("*"*50)
    #     print("trg[:-1]: ", b.trg[:, :-1])
    #     print("trg[1:]: ", b.trg) 

    #     if i == 2:
    #         break                                
