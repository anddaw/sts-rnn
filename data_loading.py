import csv
from typing import Tuple, Dict, List

import numpy as np
import torch
from sacremoses import MosesTokenizer

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


def load_labels(path: str) -> Tuple[List[torch.Tensor], List[float]]:

    def get_label(s: float):
        t = torch.zeros(6)
        t[int(round(s))] = 1
        return t

    with open(path, 'r') as labels_file:
        scores = [float(s.strip()) for s in labels_file]

    labels = [get_label(s) for s in scores]

    return labels, scores


def load_sentence_pairs(path: str) -> List[str]:

    with open(path, 'r') as sentences_file:
        return list(sentences_file)


def embed_sentences(sentences: List[str], embeddings: Embeddings) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    preprocessor = DataPreprocessor(embeddings)
    embedded = preprocessor.embed_sentence_pairs(sentences)
    print(f'Words: {preprocessor.words}, unknown: {preprocessor.unknown_words}')
    return embedded


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


