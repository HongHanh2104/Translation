from models.model import Transformer
from dictionaries import IndexDictionary
from dataset.en_vi_dataset import EN_VIDataset
from utils.utils import input_target_collate_fn, idx_to_word

import torch
from torch.utils.data import DataLoader

from argparse import ArgumentParser
import yaml
import os
import random
import numpy as np
from tqdm import tqdm

def translate(device, model, data_iter, src_dict, trg_dict):
    model.eval()
    predicts = []
    targets = []
    sources = []
    # Setup progress bar
    with torch.no_grad():
        progress_bar = tqdm(data_iter)
        for i, (src, trg) in enumerate(progress_bar):
            src = src.to(device)
            trg = trg.to(device)
            copy_trg = trg

            out = model(src, trg[:, :-1])
            trg = trg[:, 1:]

            for j in range(copy_trg.size()[0]):
                sources.append(idx_to_word(
                        x=src[j].to('cpu').numpy(),
                        vocab=src_dict.get_idx_to_words_dict()))
                targets.append(idx_to_word(
                        x=copy_trg[j].to('cpu').numpy(), 
                        vocab=trg_dict.get_idx_to_words_dict()))
                predicts.append(idx_to_word(
                        x=out[j].max(dim=1)[1].to('cpu').numpy(),
                        vocab=trg_dict.get_idx_to_words_dict())) 
        
    return sources, targets, predicts

if __name__ == '__main__':
    parser = ArgumentParser(description='TRANSLATION')
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--config')

    args = parser.parse_args()

    # Load file config
    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    # Get device
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)
    name_id = config['id']

    # Construct dictionaries
    src_dict = IndexDictionary.load(
                config['dataset']['root_dir'], 
                mode='source'
                )
    enc_voc_size = src_dict.get_vocab_size()

    trg_dict = IndexDictionary.load(
                config['dataset']['root_dir'], 
                mode='target'
                )
    dec_voc_size = trg_dict.get_vocab_size()

    src_pad_idx = 0
    trg_pad_idx = 0

    test_data = EN_VIDataset(
            data_dir=config['dataset']['root_dir'],
            phase='test')
    
    n = random.sample(range(len(test_data)), k = 5)
    samples = [] 
    for i in n:
        samples.append(test_data[i])
    
    test_dataloader = DataLoader(
            samples,
            batch_size=config['dataset']['test']['batch_size'],
            shuffle=True,
            num_workers=config['dataset']['test']['num_workers'],
            collate_fn=input_target_collate_fn,
    )


    # Build model 
    print('Building model ...')
    model = Transformer(n_src_vocab=enc_voc_size,
                        n_trg_vocab=dec_voc_size,
                        src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx,
                        max_len=config['model']['max_len'],
                        d_model=config['model']['d_model'],
                        d_ffn=config['model']['d_ffn'],
                        n_layer=config['model']['n_layer'],
                        n_head=config['model']['n_head'],
                        dropout=config['model']['dropout']
                        )
    model = model.to(device)
    model_path = os.path.join(config['pretrained_path'],
                              config['model_filename'])
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters} trainable parameters.')

    sources, targets, predicts = translate(device, model, test_dataloader, src_dict, trg_dict)
    
    for i in range(len(sources)):
        print(f'i = {i + 1}')
        print(f'Source: {sources[i]} \n')
        print(f'Target: {targets[i]} \n')
        print(f'Predict: {predicts[i]} \n')
    
          
