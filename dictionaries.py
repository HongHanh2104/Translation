from collections import Counter
import os

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
START_TOKEN = '<StartSent>'
END_TOKEN = '<EndSent>'

class IndexDictionary:
    def __init__(self, iterable=None, mode='source', vocab_size=None):
        self.special_tokens = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]
        self.mode = mode 
        
        if iterable is not None:
            self.vocab_tokens, self.token_counts = self._build_vocab(iterable, vocab_size)
            # create a dictionary with {word: counter}
            self.token_index_dict = {token: index for index, token in enumerate(self.vocab_tokens)}
            self.vocab_size = len(self.vocab_tokens)

    def token_to_index(self, token):
        try:
            return self.token_index_dict[token]
        except KeyError:
            return self.token_index_dict[UNK_TOKEN]

    def index_sentence(self, sentence):
        return [self.token_to_index(token) for token in sentence]

    def _build_vocab(self, iterable, vocab_size):
        # Count the frequence of each word in sentences.
        # Output: vocab_tokens, token_counts
        # #### vocab_tokens (list of str): word list 
        # #### token_counts (list of number): frequent counter of each corresponding word
        counter = Counter()
        for item in iterable:
            for token in item:
                counter[token] += 1
        
        if vocab_size is not None:
            most_common_vocabs = counter.most_common(vocab_size - len(self.special_tokens))
            frequent_tokens = [token for token, _ in most_common_vocabs]
            vocab_tokens = self.special_tokens + frequent_tokens
            token_counts = [0] * len(self.special_tokens) + [count for _, count in most_common_vocabs]

        else:
            all_tokens = [token for token, _ in counter.items()]
            vocab_tokens = self.special_tokens + all_tokens
            token_counts = [0] * len(self.special_tokens) + [count for _, count in counter.items()]

        return vocab_tokens, token_counts
    
    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab_dict(self):
        return self.token_index_dict

    def save(self, data_dir):
        vocab_filepath = os.path.join(data_dir, f'vocab-{self.mode}.txt')
        with open(vocab_filepath, 'w') as file:
            for vocab_index, (vocab_token, count) in enumerate(zip(self.vocab_tokens, self.token_counts)):
                file.write(str(vocab_index) + '\t' + vocab_token + '\t' + str(count) + '\n')

    @classmethod
    def load(cls, data_dir, mode, vocab_size=None):
        vocabulary_filepath = os.path.join(data_dir, f'vocab-{mode}.txt')

        vocab_tokens = {}
        token_counts = []
        with open(vocabulary_filepath) as file:
            for line in file:
                vocab_index, vocab_token, count = line.strip().split('\t')
                vocab_index = int(vocab_index)
                vocab_tokens[vocab_index] = vocab_token
                token_counts.append(int(count))

        if vocab_size is not None:
            vocab_tokens = {k: v for k, v in vocab_tokens.items() if k < vocab_size}
            token_counts = token_counts[:vocab_size]

        instance = cls(mode=mode)
        instance.vocab_tokens = vocab_tokens
        instance.token_counts = token_counts
        instance.token_index_dict = {token: index for index, token in vocab_tokens.items()}
        instance.vocab_size = len(vocab_tokens)

        return instance

if __name__ == "__main__":
    
    path = 'data/test'
    