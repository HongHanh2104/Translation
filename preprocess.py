#from text_datasets import *
from argparse import ArgumentParser
from dictionaries import IndexDictionary
import os
from vncorenlp import VnCoreNLP

import torch
from transformers import AutoModel, AutoTokenizer

START_TOKEN = '<StartSent>'
END_TOKEN = '<EndSent>'

def _get_data(root_dir, phase, limit=None):
    assert phase in ('train', 'val', 'test'), "Dataset phase must be either 'train' or 'val' or 'test'."
    
    data_name = f'raw-combined-{phase}.txt'
    
    data = []
    i = 0
    with open(os.path.join(root_dir, data_name)) as f:
        for line in f:
            src, trg = line.strip().split('\t')
            data.append((src, trg))
    
    if limit is not None:
        data = data[:limit]
    
    return data

def create_raw_src_trg_data(data_src, data_trg, phase, raw_data_dir, save_data_dir):    
    '''
    Create raw data that includes source sentences and target sentences.
    For example:
    Rachel Pike : The science behind a climate headline [\t] Khoa học đằng sau một tiêu đề về khí hậu [\n]
    
    @params:
    :data_src (txt file): source data 
    :data_trg (txt file): target data 
    '''
    
    src_path = os.path.join(raw_data_dir, phase, data_src)
    trg_path = os.path.join(raw_data_dir, phase, data_trg)

    with open(src_path) as src_file:
        src_data = src_file.readlines()
    
    with open(trg_path) as trg_file:
        trg_data = trg_file.readlines()
    
    with open(os.path.join(save_data_dir, f'raw-combined-{phase}.txt'), 'w') as file:
        for src_line, trg_line in zip(src_data, trg_data):
            if len(src_line.strip()) == 0 or len(trg_line.strip()) == 0:
                continue
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            line = f'{src_line}\t{trg_line}\n'
            file.write(line)
    print("Complete create raw-combined src-trg file")

def tokenize_data(root_dir, phase, limit=None):
    data = get_data(root_dir, phase, limit)
    tokenized_src = []
    tokenized_trg = []
    for i in range(len(data)):
        src, trg = data[i]
        tokenized_src.append(src.split())
        tokenized_trg.append(trg.split())
    return tokenized_src, tokenized_trg

def build_vocab(root_dir, save_data_dir, phase, limit=None):
    tokenized_src, tokenized_trg = tokenize_data(root_dir, phase, limit)
    src_dict = IndexDictionary(tokenized_src, mode='source')
    src_dict.save(save_data_dir)
    trg_dict = IndexDictionary(tokenized_trg, mode='target')
    trg_dict.save(save_data_dir)
    print('Complete build vocab.')
    
    return src_dict, trg_dict

def create_input_target(root_dir, phase, limit=None):
    tokenized_src, tokenized_trg = tokenize_data(root_dir, phase, limit)
    #inputs = []
    targets = []
    for i in range(len(tokenized_trg)):
        full_trg = [START_TOKEN] + tokenized_trg[i] + [END_TOKEN]
        targets.append(full_trg)
        #inputs.append(full_trg[:-1])
        #targets.append(full_trg[1:])
    
    return tokenized_src, targets

def create_index(root_dir, src_dict, trg_dict):
    join_inds = lambda indexes: ' '.join(str(index) for index in indexes)
    
    for phase in ('train', 'val'):
        srcs, trgs = create_input_target(root_dir, phase)

        with open(os.path.join(root_dir, f'indexed-{phase}.txt'), 'w') as file:
            for i in range(len(srcs)):
                indexed_srcs = join_inds(src_dict.index_sentence(srcs[i]))
                indexed_trgs = join_inds(trg_dict.index_sentence(trgs[i]))
                file.write(f'{indexed_srcs}\t{indexed_trgs}\n')
    print('Complete create index.')

def create_index_testdata(root_dir, phase):
    src_dict = IndexDictionary.load(root_dir, mode='source')
    trg_dict = IndexDictionary.load(root_dir, mode='target')
    
    join_inds = lambda indexes: ' '.join(str(index) for index in indexes)
    srcs, trgs = create_input_target(root_dir, phase)

    with open(os.path.join(root_dir, f'indexed-{phase}.txt'), 'w') as file:
        for i in range(len(srcs)):
            indexed_srcs = join_inds(src_dict.index_sentence(srcs[i]))
            indexed_trgs = join_inds(trg_dict.index_sentence(trgs[i]))
            file.write(f'{indexed_srcs}\t{indexed_trgs}\n')
    print('Complete create index.')

def get_data(root_dir, data_name, phase, limit=None):
    assert phase in ('train', 'val', 'test'), "Dataset phase must be either 'train' or 'val' or 'test'."
    
    data = []
    i = 0
    with open(os.path.join(root_dir, data_name)) as f:
        for line in f:
            data.append(line)
    
    if limit is not None:
        data = data[:limit]
    return data

def preprocess_vi_data(raw_dir, save_dir, vi_data, phase):
    rdrsegmenter = VnCoreNLP("../vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
    
    vi_path = os.path.join(raw_dir, phase, vi_data)
    
    with open(vi_path) as f:
        data = f.readlines()
    
    with open(os.path.join(save_data_dir, f'raw-vi-{phase}.txt'), 'w') as file:
        for line in data:
            result = ''
            sentences = rdrsegmenter.tokenize(line)
            for sentence in sentences:
                result += " ".join(sentence) + ' '
            save_line = f'{result}\n'
            file.write(save_line)
    print(f"Complete preproces vi {phase} data")

def tokenize_vi_data(path, phase):
    data_name = f'raw-vi-{phase}.txt'

    data = get_data(path, data_name, phase)

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

    with open(os.path.join(path, f'index-vi-{phase}.txt'), 'w') as file:
        for line in data:
            result = tokenizer.encode(line)  #(line, padding=False, max_length=1000)["input_ids"]
            save_line = f"{' '.join(list(map(str, result)))}\n"
            #save_line = f'{result}\n'
            file.write(save_line)
    print(f"Complete tokenize vi {phase} data")

def tokenize_en_data(root_dir, save_dir, data_name, phase):
    
    data = get_data(os.path.join(root_dir, phase), data_name, phase)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=False)

    with open(os.path.join(save_dir, f'index-en-{phase}.txt'), 'w') as file:
        for line in data:
            result = tokenizer.encode(line)  #(line, padding=False, max_length=1000)["input_ids"]
            save_line = f"{' '.join(list(map(str, result)))}\n"
            file.write(save_line)
    print(f"Complete tokenize en {phase} data")

if __name__ == "__main__":
    parser = ArgumentParser('Prepare Dataset')
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--en', type=str, default='train.en') # src
    parser.add_argument('--vi', type=str, default='train.vi') # trg
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--save_folder_name', type=str, default='processed-data')
    parser.add_argument('--share_dictionary', type=bool, default=False)

    args = parser.parse_args()

    raw_data_dir = os.path.join(args.root_path, 'raw-data')
    save_data_dir = os.path.join(args.root_path, args.save_folder_name)
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    # create_raw_src_trg_data(
    #         args.data_src, 
    #         args.data_trg, 
    #         args.phase,
    #         raw_data_dir, 
    #         save_data_dir)

    # src_dict, trg_dict = build_vocab(save_data_dir, save_data_dir, 'train')

    # create_index(save_data_dir, src_dict, trg_dict)
    
    #create_index_testdata(save_data_dir, phase=args.phase)
    
    # preprocess_vi_data(raw_data_dir, save_data_dir, args.vi, args.phase)
    # tokenize_vi_data(save_data_dir, args.phase)

    tokenize_en_data(raw_data_dir, save_data_dir, args.en, args.phase)
