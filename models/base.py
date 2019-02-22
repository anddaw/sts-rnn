from typing import Tuple, Any

import torch
from torch import nn
from torch.nn.modules.rnn import RNN


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, sentence_pair: Tuple[Any, Any]) -> torch.Tensor:
        raise NotImplementedError()

    def predict(self, sentence_pair: Tuple[Any, Any]) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(sentence_pair).view(self.output_size)

    def predict_score(self, sentence_pair: Tuple[Any, Any]) -> float:
        predicted_label = self.predict(sentence_pair)
        return float(sum([i * v for i, v in enumerate(predicted_label)]))
