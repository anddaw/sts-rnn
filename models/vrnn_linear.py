from typing import Tuple

import torch
from torch import nn
from torch.nn.modules.rnn import RNN


class VRNNLinear(nn.Module):

    def __init__(self, hidden_size: int, embedding_size: int, output_size: int, num_layers: int = 1):
        super(VRNNLinear, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        # left side
        self.rnn_l = RNN(input_size=embedding_size,
                         hidden_size=hidden_size,
                         bias=False,
                         num_layers=num_layers)

        self.linear_l = nn.Linear(hidden_size, hidden_size)
        self.hidden_l = self.init_hidden()

        # right side
        self.rnn_r = RNN(input_size=embedding_size,
                         hidden_size=hidden_size,
                         bias=False,
                         num_layers=num_layers)
        self.linear_r = nn.Linear(hidden_size, hidden_size)
        self.hidden_r = self.init_hidden()

        # output layer
        self.output = nn.Linear(hidden_size, output_size)

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def forward(self, sentence_pair: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        sentence_l, sentence_r = sentence_pair

        _, self.hidden_l = self.rnn_l(sentence_l.view((-1, 1, 50)))
        linear_l_out = self.linear_l(self.hidden_l[-1])

        _, self.hidden_r = self.rnn_r(sentence_r.view((-1, 1, 50)))
        linear_r_out = self.linear_r(self.hidden_r[-1])

        return torch.tanh(self.output(linear_l_out + linear_r_out))

    def predict(self, sentence_pair: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(sentence_pair).view(self.output_size)

    def predict_score(self, sentence_pair: Tuple[torch.Tensor, torch.Tensor]) -> float:
        predicted_label = self.predict(sentence_pair)
        return float(sum([i * v for i, v in enumerate(predicted_label)]))

