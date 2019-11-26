from typing import Tuple, Any
from torch import nn
import torch
import torch.nn.functional as F

from embeddings.base import Embeddings
from models.base import BaseModel


class CNN1d(BaseModel):

    def __init__(self, embeddings: Embeddings, output_size: int, sentence_length: int = 30):
        super(CNN1d, self).__init__(output_size)

        self.embeddings = embeddings
        embedding_size = embeddings.embedding_size

        self.sentence_length = sentence_length
        self.cnn_l = nn.Conv1d(1, embedding_size, embedding_size, stride=embedding_size)
        self.cnn_r = nn.Conv1d(1, embedding_size, embedding_size, stride=embedding_size)

        self.pool = nn.MaxPool1d(self.sentence_length)
        self.linear_1 = nn.Linear(embedding_size * 2, embedding_size)
        self.linear_2 = nn.Linear(embedding_size, output_size)

    def forward(self, sentence_pair: Tuple[Any, Any]) -> torch.Tensor:
        sentence_l, sentence_r = sentence_pair

        sentence_l = self.embeddings(sentence_l)
        sentence_r = self.embeddings(sentence_r)

        sentence_l = F.pad(sentence_l, (0, 0, 0, self.sentence_length-sentence_l.shape[0]))
        sentence_l = sentence_l.view(1, 1, -1)
        x_l = F.relu(self.cnn_l(sentence_l))
        x_l = self.pool(x_l)
        v_l = x_l.view(-1)

        sentence_r = F.pad(sentence_r, (0, 0, 0, self.sentence_length - sentence_r.shape[0]))
        sentence_r = sentence_r.view(1, 1, -1)
        x_r = F.relu(self.cnn_r(sentence_r))
        x_r = self.pool(x_r)
        v_r = x_r.view(-1)

        x = torch.cat((torch.abs(v_l - v_r), v_l * v_r)).view(1, -1)
        x = torch.tanh(self.linear_1(x))
        x = F.softmax(self.linear_2(x))

        return x
