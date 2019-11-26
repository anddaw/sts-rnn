from typing import List

import torch
from torch import nn


class Embeddings(nn.Module):

    def forward(self, sentence: str):
        raise NotImplementedError

    def __init__(self):
        super().__init__()
        self.embedding_size = 0
