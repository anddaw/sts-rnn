from typing import Optional

from models.base import BaseModel
from models.vrnn_linear import VRNNLinear
from models.vrnn_tree import VRNNTree


def pick_model(config) -> Optional[BaseModel]:
    num_layers = config['num_rnn_layers'] if 'num_rnn_layers' in config else 1
    if config['input_type'] == 'sentence':

        return VRNNLinear(hidden_size=50, embedding_size=50, output_size=6, num_layers=num_layers)

    elif config['input_type'] == 'tree':
        return VRNNTree(hidden_size=50, embedding_size=50, output_size=6, num_layers=num_layers)
