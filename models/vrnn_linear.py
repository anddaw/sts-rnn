from typing import Tuple, List

import torch
from torch import nn
from torch.nn.modules.rnn import RNN
import torch.nn.functional as F

from embeddings.base import Embeddings
from models.base import BaseModel


class VRNNLinear(BaseModel):

    def __init__(self, hidden_size: int, embeddings: Embeddings, output_size: int, num_layers: int = 1):
        super(VRNNLinear, self).__init__(output_size)

        self.hidden_size = hidden_size
        self.embeddings = embeddings

        # left side
        self.rnn_l = RNN(input_size=embeddings.embedding_size,
                         hidden_size=hidden_size,
                         bias=False,
                         num_layers=num_layers)

        self.linear_l = nn.Linear(hidden_size, hidden_size)

        # right side
        self.rnn_r = RNN(input_size=embeddings.embedding_size,
                         hidden_size=hidden_size,
                         bias=False,
                         num_layers=num_layers)
        self.linear_r = nn.Linear(hidden_size, hidden_size)

        # output layer
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, sentence_pair: Tuple[List[str], List[str]]) -> torch.Tensor:
        sentence_l, sentence_r = sentence_pair

        sentence_l = self.embeddings(sentence_l)
        sentence_r = self.embeddings(sentence_r)

        _, hidden_l = self.rnn_l(sentence_l.view((-1, 1, self.embeddings.embedding_size)))
        linear_l_out = self.linear_l(hidden_l[-1])

        _, hidden_r = self.rnn_r(sentence_r.view((-1, 1, self.embeddings.embedding_size)))
        linear_r_out = self.linear_r(hidden_r[-1])

        return F.softmax(self.output(torch.tanh(linear_l_out + linear_r_out)), dim=1)


