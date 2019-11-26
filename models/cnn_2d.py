from typing import Tuple, List
from torch import nn
import torch
import torch.nn.functional as F

from embeddings.base import Embeddings
from models.base import BaseModel


class CNN2d(BaseModel):

    def __init__(self, embeddings: Embeddings, output_size: int, sentence_length: int = 32):
        super(CNN2d, self).__init__(output_size)

        self.embeddings = embeddings

        self.sentence_length = sentence_length

        self.out_channels_layer_1 = 100
        self.out_channels_layer_2 = 100

        self.cnn_1 = nn.Conv2d(in_channels=embeddings.embedding_size*2, out_channels=self.out_channels_layer_1,
                               padding=1, kernel_size=3)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=4)

        self.cnn_2 = nn.Conv2d(in_channels=self.out_channels_layer_1, out_channels=self.out_channels_layer_2,
                               padding=1, kernel_size=3)

        self.classifier = nn.Linear(in_features=self.out_channels_layer_2, out_features=output_size)

    def forward(self, sentence_pair: Tuple[List[str], List[str]]) -> torch.Tensor:
        sentence_l, sentence_r = sentence_pair

        sentence_l = self.embeddings(sentence_l)
        sentence_r = self.embeddings(sentence_r)

        sentence_l = F.pad(sentence_l, (0, 0, 0, self.sentence_length-sentence_l.shape[0]))
        sentence_l = sentence_l.t()

        sentence_r = F.pad(sentence_r, (0, 0, 0, self.sentence_length - sentence_r.shape[0]))
        sentence_r = sentence_r.t()

        x_l = sentence_l.unsqueeze(1).repeat((1, self.sentence_length, 1))
        x_r = sentence_r.unsqueeze(2).repeat((1, 1, self.sentence_length))

        x = torch.cat((x_l, x_r), 0).view(1, -1, self.sentence_length, self.sentence_length)

        x = F.relu(self.cnn_1(x))
        x = self.max_pool_1(x)
        x = F.relu(self.cnn_2(x))

        x = F.max_pool2d(x, kernel_size=x.shape[3])

        x = x.view(-1, self.out_channels_layer_2)

        x = F.softmax(self.classifier(x))

        return x
