from typing import Optional

from embeddings.dict_embeddings import DictEmbeddings
from models.base import BaseModel
from models.cnn_1d import CNN1d
from models.cnn_2d import CNN2d
from models.vrnn_linear import VRNNLinear
from models.vrnn_tree import VRNNTree


def pick_model(config, output_size=6) -> Optional[BaseModel]:
    num_layers = config['num_rnn_layers'] if 'num_rnn_layers' in config else 1

    embeddings = DictEmbeddings(config['embeddings'])

    if config['model'] == 'vrnn':
        if config['input_type'] == 'sentence':

            return VRNNLinear(hidden_size=50, embeddings=embeddings, output_size=output_size,
                              num_layers=num_layers)

        elif config['input_type'] == 'tree':
            return VRNNTree(hidden_size=50, embeddings=embeddings, output_size=output_size,
                            num_layers=num_layers)

    elif config['model'] == 'cnn_1d':
        return CNN1d(embeddings=embeddings, output_size=output_size,
                     sentence_length=config['sentence_length'])

    elif config['model'] == 'cnn_2d':

        return CNN2d(embeddings=embeddings, output_size=output_size, sentence_length=config['sentence_length'])

