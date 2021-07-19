from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=['data/en-vi/raw-data/train/train.en',
           'data/en-vi/raw-data/val/tst2012.en'],
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)

tokenizer.save_model('.', 'english')
