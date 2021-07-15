from models import Transformer
from text_datasets import IndexedInputTargetTranslationDataset
from dictionaries import IndexDictionary

import torch

from argparse import ArgumentParser
import json
import yaml
import os

def preprocess(seq_dict):
    return IndexedInputTargetTranslationDataset.preprocess(seq_dict)

def translate(seq,
              src_dict,
              n,
              config,
              pretrain_path):

    src = preprocess(src_dict)(seq)
    src_tensor = torch.tensor(src).unsqueeze(0)
    len_tensor = torch.tensor(len(src_tensor)).unsqueeze(0)

    src_mask = 
    

if __name__ == '__main__':
    parser = ArgumentParser(description='TRANSLATION')
    parser.add_argument('--src', type=str)
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--config')
    parser.add_argument('--pretrained_path')

    args = parser.parse_args()
    print(args.src)
    # Load file config
    config_path = os.path.join('./configs', args.config + '.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    # Get device
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)
    name_id = config['id']

    # Construct dictionaries
    src_dict = IndexDictionary.load(config['dataset']['root_dir'], mode='source')
    trg_dict = IndexDictionary.load(config['dataset']['root_dir'], mode='target')

    # Build model 
    print("Building model .....")
    model = Transformer(src_dict.vocab_size, 
                        trg_dict.vocab_size,
                        config['model']['pad_index'],
                        config['model']['pad_index'],
                        config['model']['d_model'],
                        config['model']['n_layer'],
                        config['model']['n_head'],
                        config['model']['n_position'])
    model = model.to(device)
    
    translate(args.src, src_dict, model, config, args.pretrained_path)

    
          
