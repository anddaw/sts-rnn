import torch
from typing import Tuple, List

from sacremoses import MosesTokenizer


class DataPreprocessor:

    def __init__(self):

        self.words = 0
        self.unknown_words = 0

        self.tokenizer = MosesTokenizer()

    def preprocess_sentence_pairs(self, sentence_pairs: List[str]) -> List[Tuple[List[str], List[str]]]:

        preprocessed = []

        def preprocess_sentence(sentence: str) -> List[str]:
            return self.tokenizer.tokenize(sentence.strip())

        for sentence_pair in sentence_pairs:

            sentence1, sentence2 = sentence_pair.strip().split('\t')

            preprocessed.append((preprocess_sentence(sentence1), preprocess_sentence(sentence2)))

        return preprocessed


