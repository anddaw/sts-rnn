from torchtext.vocab import Vocab
from collections import Counter


class Tokenizer:

    def __init__(self):
        self.vocab = Vocab(Counter())

    def fit(self, sentences):
        self.vocab = Vocab(
            Counter(w for s in self.tokenize(sentences) for w in s)
        )

    def sentences_to_ids(self, sentences):
        for sentence in self.tokenize(sentences):
            yield [
                self.vocab.stoi.get(w, self.vocab.stoi[self.vocab.UNK]) for w in sentence
            ]

    @staticmethod
    def tokenize(ss):
        return (s.split() for s in ss)
