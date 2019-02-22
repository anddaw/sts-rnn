from typing import Optional

from models.base import BaseModel
from models.vrnn_linear import VRNNLinear


def pick_model(config) -> Optional[BaseModel]:

    if config['input_type'] == 'sentence':
        num_layers = config['num_rnn_layers'] if 'num_rnn_layers' in config else 1
        return VRNNLinear(hidden_size=50, embedding_size=50, output_size=6, num_layers=num_layers)
