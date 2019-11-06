import csv
from typing import Dict, Tuple

import numpy as np
import torch

from embeddings.base import Embeddings
from utils import log

UNKNOWN_TOKEN = '*UNKNOWN*'
EmbeddingsDict = Tuple[Dict[str, int], np.ndarray]


class DictEmbeddings(Embeddings):

    def _word_index(self, word):
        return self.dictionary.get(word, self.dictionary[UNKNOWN_TOKEN])

    def __init__(self, path: str):

        log(f'Loading embeddings from {path}')
        embeddings = []
        dictionary = {}
        with open(path, 'r') as embeddings_file:
            reader = csv.reader(embeddings_file, delimiter=' ', quoting=csv.QUOTE_NONE)

            for i, row in enumerate(reader):
                dictionary[row[0]] = i
                embeddings.append([float(s) for s in row[1:]])

        self.dictionary = dictionary
        self.embedding_vectors = torch.FloatTensor(embeddings)

        self.embedding_size = len(self.embedding_vectors[0])

    def embed(self, word: str) -> torch.Tensor:
        return self.embedding_vectors[self._word_index(word)]


