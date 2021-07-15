from collections import Counter
import math
import numpy as np
import os 
import random

import torch

from model.models import Transformer
from dataset.multi30k import Multi30kLoader
from metrics import BLEUMetric
from utils.utils import idx_to_word

import argparse 
import yaml

def test(model_filename, config, n, result_dir='./results'):
    # Get device
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Load test data
    print("Loading dataset")
    data = Multi30kLoader(ext=('.en', '.de'))
    train, _, test = data.create_dataset()
    data.build_vocab(data=train, min_freq=2)
    _, _, test_iter = data.make_iter(
                            batch_size=config['dataset']['train']['batch_size'],
                            device=device
                        )
    src_pad_idx, trg_pad_idx = data.get_pad_idx()
    enc_voc_size, dec_voc_size = data.get_voc_size()


    # Load trained model
    print("Loading model")
    path = os.path.join('./checkpoints', model_filename)
    model = Transformer(n_src_vocab=enc_voc_size, 
                        n_trg_vocab=dec_voc_size,
                        src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx,
                        max_len=config['model']['max_len'],
                        d_model=config['model']['d_model'],
                        d_ffn=config['model']['d_ffn'],
                        n_layer=config['model']['n_layer'],
                        n_head=config['model']['n_head'],
    )
    model = model.to(device)
    model.load_state_dict(torch.load(path)['model_state_dict'])

    # Get metric BLEU
    metric = BLEUMetric()


    with torch.no_grad():
        batch_bleu = []
        result_path = os.path.join(result_dir, 'test-1.33' + '.txt')
        file = open(result_path, 'w')
        for i, batch in enumerate(test_iter):
            
            src = batch.src
            trg = batch.trg
            out = model(src, trg[:, :-1])

            total_bleu = []
            
            for j in range(n):
                try: 
                    #print(i, src.size())
                    src_words = idx_to_word(src[j], data.source.vocab)
                    trg_words = idx_to_word(trg[j], data.target.vocab)
                    out_words = out[j].max(dim=1)[1]
                    out_words = idx_to_word(out_words, data.target.vocab)
                    
                    
                    file.write(f'+++++++++++++++ PREDICTION RESULT +++++++++++++++ \n')
                    file.write(f'{i}: \n')
                    file.write(f'Source: {src_words} \n')
                    file.write(f'Target: {trg_words} \n')
                    file.write(f'Predicted word: {out_words} \n')
                    file.write(f'******************************************************\n')

                    blue = metric.get_bleu(hypothesis=out_words.split(), reference=trg_words.split())
                    total_bleu.append(blue)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            file.write(f'BLEU score = {total_bleu} \n')
            file.write(f'******************************************************\n')
            batch_bleu.append(total_bleu)

        batch_bleu = sum(batch_bleu) / len(batch_bleu)
        file.write(f'TOTAL BLEU SCORE = {batch_bleu}') 
        file.close()          
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--config')
    parser.add_argument('--n', type=int)
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    test(args.model, config, args.n)

    
