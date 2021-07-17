#from text_datasets import *
from argparse import ArgumentParser
from dictionaries import IndexDictionary
import os

START_TOKEN = '<StartSent>'
END_TOKEN = '<EndSent>'

def get_data(root_dir, phase, limit=None):
    assert phase in ('train', 'val'), "Dataset phase must be either 'train' or 'val'."
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

def create_raw_src_trg_data(train_src, train_trg, val_src, val_trg, raw_data_dir, save_data_dir):    
    '''
    Create raw data that includes source sentences and target sentences.
    For example:
    Rachel Pike : The science behind a climate headline [\t] Khoa học đằng sau một tiêu đề về khí hậu [\n]
    
    @params:
    :train_src (txt file): source data for train
    :train_trg (txt file): target data for train
    :val_src (txt file): source data for val
    :val_trg (txt file): target data for val
    '''
    
    file_map = {
        'train': [train_src, train_trg],
        'val': [val_src, val_trg]
    }
    for phase in ('train', 'val'):
        src_path = os.path.join(raw_data_dir, phase, file_map[phase][0])
        trg_path = os.path.join(raw_data_dir, phase, file_map[phase][1])

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
                #indexed_inps = join_inds(trg_dict.index_sentence(inps[i]))
                indexed_trgs = join_inds(trg_dict.index_sentence(trgs[i]))
                #file.write(f'{indexed_srcs}\t{indexed_inps}\t{indexed_trgs}\n')
                file.write(f'{indexed_srcs}\t{indexed_trgs}\n')
    print('Complete create index.')

if __name__ == "__main__":
    parser = ArgumentParser('Prepare Dataset')
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--train_src', type=str, default='train.en')
    parser.add_argument('--train_trg', type=str, default='train.vi')
    parser.add_argument('--val_src', type=str, default='tst2012.en')
    parser.add_argument('--val_trg', type=str, default='tst2012.vi')
    parser.add_argument('--save_folder_name', type=str, default='processed-data')
    parser.add_argument('--share_dictionary', type=bool, default=False)

    args = parser.parse_args()

    raw_data_dir = os.path.join(args.root_path, 'raw-data')
    save_data_dir = os.path.join(args.root_path, args.save_folder_name)
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    # create_raw_src_trg_data(
    #         args.train_src, 
    #         args.train_trg, 
    #         args.val_src, 
    #         args.val_trg,
    #         raw_data_dir, 
    #         save_data_dir)

    # train_data = get_data(save_data_dir, 'train')
    # val_data = get_data(save_data_dir, 'val')

    src_dict, trg_dict = build_vocab(save_data_dir, save_data_dir, 'train')

    create_index(save_data_dir, src_dict, trg_dict)
    
    # tokenized_dataset = TokenizedTranslationDataset(args.save_data_dir, 'train')
    # #print(tokenized_dataset[0])
    # src_generator = source_tokens_generator(tokenized_dataset)
    # src_dict = IndexDictionary(src_generator, mode='source')
    # trg_generator = target_tokens_generator(tokenized_dataset)
    # trg_dict = IndexDictionary(trg_generator, mode='target')
    # src_dict.save(args.save_data_dir)
    # trg_dict.save(args.save_data_dir)

    # IndexedInputTargetTranslationDataset.prepare(args.save_data_dir, src_dict, trg_dict)

    # print('Done dataset preparation')


