from models.model import Transformer
from dataset.multi30k import Multi30kLoader
from losses import TokenCrossEntropyLoss
from metrics import BLEUMetric
from trainer import Trainer

import torch
import torch.nn as nn
from torch import optim
from torch.optim import Adam 

import argparse
from datetime import datetime
import random
import yaml
import argparse

def train(config):
    # Get device
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)
    
    # Build dataset
    random.seed(config['seed'])
    print('Building dataset ...')
    data = Multi30kLoader(ext=('.en', '.de'))
    train, val, _ = data.create_dataset()
    data.build_vocab(data=train, min_freq=2)
    train_iter, val_iter, _ = data.make_iter(
                                        batch_size=config['dataset']['train']['batch_size'],
                                        device=device
                                        )
    src_pad_idx, trg_pad_idx = data.get_pad_idx()
    enc_voc_size, dec_voc_size = data.get_voc_size()

    # Define model
    random.seed(config['seed'])
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
        
    # Get pretrained model
    pretrained_path = config['pretrained_path']
    pretrained = None
    if pretrained_path != None:
        pretrained = torch.load(pretrained_path, map_location=dev_id)
        for item in ["model"]:
            config[item] = pretrained["config"][item]

    # Train from pretrained if it is not None
    if pretrained is not None:
        model.load_state_dict(pretrained['model_state_dict'])
    
    model = model.to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters} trainable parameters.')

    # Define loss
    random.seed(config['seed'])
    #loss = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
    loss = TokenCrossEntropyLoss()
    loss = loss.to(device)

    # Define metrics
    random.seed(config['seed'])
    metric = BLEUMetric()

    # Define Optimizer 
    random.seed(config['seed'])
    optimizer = Adam(model.parameters(), 
                    lr=config['trainer']['lr'],
                )
                    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer=optimizer,
                            verbose=True,
                            factor=config['optimizer']['factor'],
                            patience=config['optimizer']['patience']
                        )

    print('Start training ...')

    # Define trainer
    random.seed(config['seed'])
    trainer = Trainer(model=model,
                      device=device,
                      data=data,
                      iterator=(train_iter, val_iter),
                      loss=loss,
                      metric=metric,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      config=config
                    )
    
    # Start to train 
    random.seed(config['seed'])
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    train(config)