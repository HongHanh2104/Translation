# Neural Machine Translation using different Positional Encoding methods

## How to use

### Tokenization

To train a tokenizer, use `train_tokenize.py`.

```
usage: train_tokenize.py [-h] [-t {wordpiece,bpe}] [-l] [-f FILES [FILES ...]] [-m MIN_FREQ] [-o OUT_DIR] [-of OUT_FN]

Train a tokenizer

optional arguments:
  -h, --help            show this help message and exit
  -t {wordpiece,bpe}, --token_type {wordpiece,bpe}
                        Tokenization type
  -l, --lowercase       Toggle using lowercase
  -f FILES [FILES ...], --files FILES [FILES ...]
                        List of files to train tokenizers
  -m MIN_FREQ, --min_freq MIN_FREQ
                        Minimum number of occurrences for a token to be included
  -o OUT_DIR, --out_dir OUT_DIR
                        Output directory
  -of OUT_FN, --out_fn OUT_FN
                        Output filename
```

For example,
```
python train_tokenize.py \
    --token_type wordpiece \
    --files data/en-vi/raw-data/train/train.en \
    --min_freq 5
```

### Training

To train a model, use `train.py`.

```
usage: train.py [-h] [--config CONFIG]

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  Path to configuration file
```

For example,

```
python train.py \
    --config configs/translation_config.yaml
```

Detail about configuration can be found in `configs/translation_config.yaml`. Most of the hyperparameters' names are self-explained.

### Evaluation

To evaluate a trained model, use `eval.py`.

```
usage: eval.py [-h] [--cuda] [-w WEIGHTS] [-s SOURCE] [-t TARGET] [-bs BATCH_SIZE]

Evaluate a model.

optional arguments:
  -h, --help            show this help message and exit
  --cuda                Toggle using GPU
  -w WEIGHTS, --weights WEIGHTS
                        Path to pretrained weights
  -s SOURCE, --source SOURCE
                        Source language validation file
  -t TARGET, --target TARGET
                        Target language validation file
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Size of batches
```

For example
```
python eval.py \
    --weights weights/Transformer-En2Vi-VPE-WordPiece-ShareEmbPrj/best_bleu.pth \
    --source data/en-vi/raw-data/test/tst2013.en \
    --target data/en-vi/raw-data/test/tst2013.vi \
    --cuda
```

### Inference

To use a CLI inference, use `infer.py`.

```
usage: infer.py [-h] [--cuda] [-w WEIGHTS]

CLI inference

optional arguments:
  -h, --help            show this help message and exit
  --cuda                Toggle using GPU
  -w WEIGHTS, --weights WEIGHTS
                        Path to pretrained weights
```

For example
```
python infer.py \
    --weights weights/Transformer-En2Vi-VPE-WordPiece-ShareEmbPrj/best_bleu.pth \
    --cuda
```

An example,
```
Q: I want to be a doctor.
A: Tôi muốn trở thành bác sĩ.
Q: My name is Thomas.
A: Tôi tên là Thomas.
Q: My love is for you and you only.
A: Tôi yêu bạn và chỉ có bạn thôi.
Q: When you lack knowledge, dig deep to find it.
A: Khi bạn thiếu hiểu biết, bạn phải tìm kiếm nó.
```