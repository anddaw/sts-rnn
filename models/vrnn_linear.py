from typing import Tuple

import torch
from torch import nn
from torch.nn.modules.rnn import RNN
import torch.nn.functional as F

from models.base import BaseModel


class VRNNLinear(BaseModel):

    def __init__(self, hidden_size: int, embedding_size: int, output_size: int, num_layers: int = 1):
        super(VRNNLinear, self).__init__(output_size)

        self.hidden_size = hidden_size

        # left side
        self.rnn_l = RNN(input_size=embedding_size,
                         hidden_size=hidden_size,
                         bias=False,
                         num_layers=num_layers)

        self.linear_l = nn.Linear(hidden_size, hidden_size)

        # right side
        self.rnn_r = RNN(input_size=embedding_size,
                         hidden_size=hidden_size,
                         bias=False,
                         num_layers=num_layers)
        self.linear_r = nn.Linear(hidden_size, hidden_size)

        # output layer
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, sentence_pair: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        sentence_l, sentence_r = sentence_pair

        _, hidden_l = self.rnn_l(sentence_l.view((-1, 1, 50)))
        linear_l_out = self.linear_l(hidden_l[-1])

        _, hidden_r = self.rnn_r(sentence_r.view((-1, 1, 50)))
        linear_r_out = self.linear_r(hidden_r[-1])

        return F.softmax(self.output(torch.tanh(linear_l_out + linear_r_out)), dim=1)


