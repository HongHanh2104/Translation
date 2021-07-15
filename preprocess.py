from text_datasets import *
from argparse import ArgumentParser
from dictionaries import IndexDictionary
import os


def source_tokens_generator(dataset):
    for source, target in dataset:
        for token in source:
            yield token

def target_tokens_generator(dataset):
    for source, target in dataset:
        for token in target:
            yield token

def prepare_data(train_src, train_trg, val_src, val_trg, save_data_dir):
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)
    
    for phase in ('train', 'val'):
        if phase == 'train':
            src_filepath = train_src
            trg_filepath = train_trg
        else:
            src_filepath = val_src
            trg_filepath = val_trg
        
        with open(src_filepath) as src_file:
            src_data = src_file.readlines()

        with open(trg_filepath) as trg_file:
            trg_data = trg_file.readlines()

        with open(os.path.join(save_data_dir, f'raw-{phase}.txt'), 'w') as file:
            for src_line, trg_line in zip(src_data, trg_data):
                src_line = src_line.strip()
                trg_line = trg_line.strip()
                line = f'{src_line}\t{trg_line}\n'
                file.write(line)



if __name__ == "__main__":
    parser = ArgumentParser('Prepare Dataset')
    parser.add_argument('--train_src', type=str, default='data/test-src-train.txt')
    parser.add_argument('--train_trg', type=str, default='data/test-trg-train.txt')
    parser.add_argument('--val_src', type=str, default='data/test-src-val.txt')
    parser.add_argument('--val_trg', type=str, default='data/test-trg-val.txt')
    parser.add_argument('--save_data_dir', type=str, default='data/test')
    parser.add_argument('--share_dictionary', type=bool, default=False)

    args = parser.parse_args()

    prepare_data(args.train_src, 
             args.train_trg, 
             args.val_src, 
             args.val_trg, 
             args.save_data_dir)

    tokenized_dataset = TokenizedTranslationDataset(args.save_data_dir, 'train')
    #print(tokenized_dataset[0])
    src_generator = source_tokens_generator(tokenized_dataset)
    src_dict = IndexDictionary(src_generator, mode='source')
    trg_generator = target_tokens_generator(tokenized_dataset)
    trg_dict = IndexDictionary(trg_generator, mode='target')
    src_dict.save(args.save_data_dir)
    trg_dict.save(args.save_data_dir)

    IndexedInputTargetTranslationDataset.prepare(args.save_data_dir, src_dict, trg_dict)

    print('Done dataset preparation')


