from typing import Tuple

import torch
from torch import nn
from torch.nn.modules.rnn import RNN
import torch.nn.functional as F

from embeddings.base import Embeddings
from models.base import BaseModel

from tree import TreeNode


class VRNNTree(BaseModel):

    def __init__(self, hidden_size: int, embeddings: Embeddings, output_size: int, num_layers: int = 1):
        super(VRNNTree, self).__init__(output_size)

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

    def _forward_node(self, node: TreeNode, rnn: RNN) -> torch.Tensor:
        embedding = self.embeddings(node.word).mean(axis=0).view(1, -1)

        if node.children:
            prev_hidden = sum([self._forward_node(c, rnn) for c in node.children])
            _, hidden = rnn(torch.zeros(1, 1, self.embeddings.embedding_size), prev_hidden)
        else:
            _, hidden = rnn(embedding.view((1, 1, self.embeddings.embedding_size)))

        return hidden

    def forward(self, sentence_pair: Tuple[TreeNode, TreeNode]) -> torch.Tensor:
        tree_l, tree_r = sentence_pair

        hidden_l = self._forward_node(tree_l, self.rnn_l)
        linear_l_out = self.linear_l(hidden_l[-1])

        hidden_r = self._forward_node(tree_r, self.rnn_r)
        linear_r_out = self.linear_r(hidden_r[-1])

        return F.softmax(self.output(torch.tanh(linear_l_out + linear_r_out)), dim=1)


