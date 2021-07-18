from dataset.multi30k import Multi30kLoader
from dataset.en_vi_dataset import EN_VIDataset
from dictionaries import IndexDictionary
from torchtext.legacy.data import BucketIterator
from utils.utils import input_target_collate_fn, idx_to_word

from torch.utils.data import DataLoader

if __name__ == '__main__':
    #data = Multi30kLoader(ext=('.en', '.de'))
    data_dir = './data/en-vi/raw-data'

    # print('Building dictionaries ...')
    # src_dict = IndexDictionary.load(
    #                 data_dir=data_dir,
    #                 mode='source')
    # print(f'Source vocab size: {src_dict.get_vocab_size()}')

    # #print(src_dict.token_index_dict)

    # trg_dict = IndexDictionary.load(
    #                 data_dir=data_dir,
    #                 mode='target')
    # print(f'Target vocab size: {trg_dict.get_vocab_size()}')
    
    # test_data = EN_VIDataset(
    #                 data_dir= data_dir,
    #                 phase='test')

    # trg = test_data[0][1] 
    # trg = trg[1:]
    # #print(trg_dict.get_idx_to_words_dict())
    
    # words = idx_to_word(x=trg, vocab=trg_dict.get_idx_to_words_dict())
    # print(words)

    train_data = EN_VIDataset(
                    data_dir= data_dir,
                    phase='train')
    print(train_data[0])
    
    
    # train_dataloader = DataLoader(
    #                 train_data,
    #                 batch_size=8,
    #                 shuffle=True,
    #                 collate_fn=input_target_collate_fn
    #                 )

   
    # for i, (srcs, trgs) in enumerate(train_iter):
    #     srcs.to('cuda')
    #     trgs.to('cuda')
    #     break
    #       break
    #       #print("src: ", b.src)
    #     print("*"*50)
    #     print("trg[:-1]: ", b.trg[:, :-1])
    #     print("trg[1:]: ", b.trg) 

    #     if i == 2:
    #         break                                
