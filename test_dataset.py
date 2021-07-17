from dataset.multi30k import Multi30kLoader
from dataset.en_vi_dataset import EN_VIDataset
from dictionaries import IndexDictionary
from utils.utils import input_target_collate_fn

from torch.utils.data import DataLoader

if __name__ == '__main__':
    #data = Multi30kLoader(ext=('.en', '.de'))
    data_dir = './data/en-vi/processed-data'

    print('Building dictionaries ...')
    src_dict = IndexDictionary.load(
                    data_dir=data_dir,
                    mode='source')
    print(f'Source vocab size: {src_dict.get_vocab_size()}')
    trg_dict = IndexDictionary.load(
                    data_dir=data_dir,
                    mode='target')
    print(f'Target vocab size: {trg_dict.get_vocab_size()}')
    
    train_data = EN_VIDataset(
                    data_dir= data_dir,
                    phase='train')
    
    # val_data = EN_VIDataset(
    #                 data_dir= data_dir,
    #                 phase='val')
    
    train_dataloader = DataLoader(
                    train_data,
                    batch_size=8,
                    shuffle=True,
                    collate_fn=input_target_collate_fn
                    )

    # print(data[0])
    
    #print(enc_voc_size, dec_voc_size)

    for i, (srcs, trgs) in enumerate(train_dataloader):
          srcs.to('cuda')
          trgs.to('cuda')
          
          break
          #print("src: ", b.src)
    #     print("*"*50)
    #     print("trg[:-1]: ", b.trg[:, :-1])
    #     print("trg[1:]: ", b.trg) 

    #     if i == 2:
    #         break                                
