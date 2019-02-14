import csv
from typing import Tuple, Dict, List

import numpy as np
import torch

Embeddings = Tuple[Dict[str, int], np.ndarray]

UNKNOWN_TOKEN = '*UNKNOWN*'


def load_embeddings(path: str) -> Embeddings:

    embeddings = []
    dictionary = {}
    with open(path, 'r') as embeddings_file:
        reader = csv.reader(embeddings_file, delimiter=' ', quoting=csv.QUOTE_NONE)

        for i, row in enumerate(reader):
            dictionary[row[0]] = i
            embeddings.append([float(s) for s in row[1:]])

    return dictionary, np.asarray(embeddings)


def load_labels(path:str) -> List[torch.Tensor]:

    def get_label(s: str):
        t = torch.zeros(6)
        t[int(round(float(s.strip())))] = 1
        return t

    with open(path, 'r') as labels_file:
        return [get_label(l) for l in labels_file]


def load_sentence_pairs(path:str) -> List[str]:

    with open(path, 'r') as sentences_file:
        return list(sentences_file)


def embed_sentences(sentences: List[str], embeddings: Embeddings) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    preprocessor = DataPreprocessor(embeddings)
    return preprocessor.embed_sentence_pairs(sentences)


class DataPreprocessor:

    def __init__(self, embeddings: Embeddings):
        self.dictionary, self.embedding_vectors = embeddings

    def _word_index(self, word):
        return self.dictionary.get(word, self.dictionary[UNKNOWN_TOKEN])

    def embed_sentence_pairs(self, sentence_pairs: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor]]:

        embedded = []

        def embed_sentence(sentence: str) -> torch.Tensor:
            words = sentence.strip().split()
            return torch.tensor([self.embedding_vectors[self._word_index(w)] for w in words])

        for sentence_pair in sentence_pairs:

            sentence1, sentence2 = sentence_pair.strip().split('\t')


            embedded.append((embed_sentence(sentence1), embed_sentence(sentence2)))

        return embedded


