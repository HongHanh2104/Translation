from models.model import Transformer
from complex_models.model import ComplexTransformer
from dataset.en_vi_dataset import EN_VIDataset
from utils.utils import input_target_collate_fn
from losses import TokenCrossEntropyLoss
from trainer import Trainer
from optimizers import ScheduledOptim

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import Adam, optimizer
import nltk

import argparse
import random
import yaml


def train(config):
    # Get device
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)

    # Build dataset
    random.seed(config['seed'])
    # print('Building dataset ...')
    # data = Multi30kLoader(ext=('.en', '.de'))
    # train, val, _ = data.create_dataset()
    # data.build_vocab(data=train, min_freq=2)
    # train_iter, val_iter, _ = data.make_iter(
    #     batch_size=config['dataset']['train']['batch_size'],
    #     device=device
    # )
    # src_pad_idx, trg_pad_idx = data.get_pad_idx()
    # enc_voc_size, dec_voc_size = data.get_voc_size()

    # print('Building vocabularies ...')
    # src_dict = IndexDictionary.load(
    #     data_dir=config['dataset']['root_dir'],
    #     mode='source')
    # enc_voc_size = src_dict.get_vocab_size()
    # print(f'Source vocab size: {enc_voc_size}')

    # trg_dict = IndexDictionary.load(
    #     data_dir=config['dataset']['root_dir'],
    #     mode='target')
    # dec_voc_size = trg_dict.get_vocab_size()
    # print(f'Target vocab size: {dec_voc_size}')

    print('Building dataset ...')
    train_data = EN_VIDataset(**config['dataset']['train']['config'])
    val_data = EN_VIDataset(**config['dataset']['val']['config'])

    train_dataloader = DataLoader(
        train_data,
        batch_size=config['dataset']['train']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['train']['num_workers'],
        collate_fn=input_target_collate_fn,
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=config['dataset']['val']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['val']['num_workers'],
        collate_fn=input_target_collate_fn,
    )
    src_pad_idx = 0
    trg_pad_idx = 0

    # Define model
    random.seed(config['seed'])
    print('Building model ...')
    model = Transformer(
        n_src_vocab=train_data.en_tokenizer.get_vocab_size(),
        n_trg_vocab=train_data.vi_tokenizer.get_vocab_size(),
        src_pad_idx=train_data.en_tokenizer.token_to_id('<pad>'),
        trg_pad_idx=train_data.vi_tokenizer.token_to_id('<pad>'),
        **config['model']
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
    # loss = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
    loss = TokenCrossEntropyLoss(
        pad_idx=train_data.vi_tokenizer.token_to_id('<pad>'))
    loss = loss.to(device)

    # Define metrics
    random.seed(config['seed'])
    # metric = BLEUMetric()
    metric = nltk.translate.bleu_score

    # Define Optimizer
    random.seed(config['seed'])
    optimizer = Adam(
        model.parameters(),
        lr=config['trainer']['lr'],
        betas=(0.9, 0.98), eps=1e-09
    )
    # optimizer = ScheduledOptim(
    # params=filter(lambda x: x.requires_grad, model.parameters()),
    # betas=(0.9, 0.98), eps=1e-09,
    # d_model=512, n_warmup_steps=4000)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        verbose=True,
        factor=config['optimizer']['factor'],
        patience=config['optimizer']['patience']
    )

    print('Start training ...')

    # Define trainer
    random.seed(config['seed'])
    trainer = Trainer(
        model=model,
        device=device,
        iterator=(train_dataloader, val_dataloader),
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
