import torch
from typing import Tuple, Dict, List

import numpy as np
from sacremoses import MosesTokenizer


UNKNOWN_TOKEN = '*UNKNOWN*'
Embeddings = Tuple[Dict[str, int], np.ndarray]


class DataPreprocessor:

    def __init__(self, embeddings: Embeddings):
        self.dictionary, self.embedding_vectors = embeddings
        self.words = 0
        self.unknown_words = 0

        self.tokenizer = MosesTokenizer()

    def _word_index(self, word):
        return self.dictionary.get(word, self.dictionary[UNKNOWN_TOKEN])

    def embed_sentence_pairs(self, sentence_pairs: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor]]:

        embedded = []

        def embed_sentence(sentence: str) -> torch.Tensor:
            words = self.tokenizer.tokenize(sentence.strip())
            indices = [self._word_index(w) for w in words]
            self.words += len(indices)
            self.unknown_words += len([i for i in indices if i == self.dictionary[UNKNOWN_TOKEN]])

            return torch.FloatTensor([self.embedding_vectors[i] for i in indices])

        for sentence_pair in sentence_pairs:

            sentence1, sentence2 = sentence_pair.strip().split('\t')

            embedded.append((embed_sentence(sentence1), embed_sentence(sentence2)))

        return embedded

    def embed(self, word: str) -> torch.Tensor:
        return self.embedding_vectors[self._word_index(word)]
