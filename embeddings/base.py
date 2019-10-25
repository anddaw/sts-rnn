from typing import List

import torch


class Embeddings:

    def __init__(self):
        self.embedding_size = 0

    def embed(self, word: str) -> torch.Tensor:
        raise NotImplementedError()

    def embed_sentence(self, sentence: List[str]) -> torch.Tensor:

        s = [self.embed(w) for w in sentence]
        return torch.stack(s)