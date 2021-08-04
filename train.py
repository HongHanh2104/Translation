from src.dataset.en_vi_dataset import EN_VIDataset
from src.models.model import Transformer
from src.complex_models.model import ComplexTransformer
from src.utils.utils import input_target_collate_fn
from src.losses import TokenCrossEntropyLoss
from src.trainer import Trainer
from src.optimizers import ScheduledOptim

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import Adam, optimizer
import nltk
import yaml

import argparse
import random


def train(config):
    # Get device
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)

    # Build dataset
    random.seed(config['seed'])

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
    loss = TokenCrossEntropyLoss(
        pad_idx=train_data.vi_tokenizer.token_to_id('<pad>'))
    loss = loss.to(device)

    # Define metrics
    random.seed(config['seed'])
    metric = nltk.translate.bleu_score

    # Define Optimizer
    random.seed(config['seed'])
    optimizer = Adam(
        model.parameters(),
        lr=config['trainer']['lr'],
        betas=(0.9, 0.98), eps=1e-09
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
    parser.add_argument('--config', help='Path to configuration file')
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    train(config)
