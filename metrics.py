import math
import numpy as np
from collections import Counter

class AccuracyMetric:
    def __init__(self, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
    
    def calculate(self, outputs, targets):
        #print('metric_out: ', outputs.size())
        #print('metric_trg: ', targets.size())
        batch_size, seq_len, vocab_size = outputs.size()

        outputs = outputs.view(batch_size * seq_len, vocab_size)
        targets = targets.view(batch_size * seq_len)

        predicts = outputs.argmax(dim=1)
        correct_count = predicts == targets 

        correct_count.masked_fill_((targets == self.pad_idx), 0)

        n_word_correct = correct_count.sum().item()
        # print(correct_count)
        n_word_total = (targets != self.pad_idx).sum().item()
        
        return n_word_correct, n_word_total

class BLEUMetric:
    def __init__(self):
        super().__init__()
    
    def _computer_bleu_stats(self, hypothesis, reference):
        """
        Compute statistics for BLEU.
        """
        stats = []
        stats.append(len(hypothesis))
        stats.append(len(reference))

        for n in range(1, 5):
            s_ngrams = Counter(
                [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
            )

            r_ngrams = Counter(
                [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
            )

            stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
            stats.append(max([len(hypothesis) + 1 - n, 0]))

        return stats
    
    def _bleu(self, stats):
        """
        Computer BLEU given n-gram statistics.
        """
        if len(list(filter(lambda x: x == 0, stats))) > 0:
            return 0
        
        (c, r) = stats[:2]
        log_bleu_prec = sum(
            [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
        ) / 4.

        return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)
    
    def get_bleu(self, hypothesis, reference):
        """
        Get validation BLEU score for dev set.
        """
        stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        for hyp, ref in zip(hypothesis, reference):
            stats += np.array(self._computer_bleu_stats(hyp, ref))
        return 100 * self._bleu(stats)
    
    def idx_to_word(self, x, vocab):
        words = []
        
        for i in x:
            word = vocab.itos[i]
            if '<' not in word:
                words.append(word)
        words = " ".join(words)
        return words
