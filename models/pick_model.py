from typing import Optional

from models.base import BaseModel
from models.cnn_1d import CNN1d
from models.cnn_2d_densenet import CNN2dDenseNet
from models.vrnn_linear import VRNNLinear
from models.vrnn_tree import VRNNTree


def pick_model(config) -> Optional[BaseModel]:
    num_layers = config['num_rnn_layers'] if 'num_rnn_layers' in config else 1
    if config['model'] == 'vrnn':
        if config['input_type'] == 'sentence':

            return VRNNLinear(hidden_size=50, embedding_size=50, output_size=6, num_layers=num_layers)

        elif config['input_type'] == 'tree':
            return VRNNTree(hidden_size=50, embedding_size=50, output_size=6, num_layers=num_layers)

    elif config['model'] == 'cnn_1d':
        return CNN1d(embedding_size=50, output_size=6, sentence_length=config['sentence_length'])

    elif config['model'] == 'cnn_2d_densenet':

        return CNN2dDenseNet(embedding_size=50, output_size=6, sentence_length=config['sentence_length'])

