from typing import Optional

from embeddings.bert_embeddings import BertEmbeddings
from embeddings.cached_embeddings import CachedEmbeddings
from embeddings.dict_embeddings import DictEmbeddings
from models.base import BaseModel
from models.cnn_1d import CNN1d
from models.cnn_2d import CNN2d
from models.vrnn_linear import VRNNLinear
from models.vrnn_tree import VRNNTree


def pick_model(config, output_size=6) -> Optional[BaseModel]:
    num_layers = config.get('num_rnn_layers', 1)

    if config['embeddings'] == 'dict':
        embeddings = DictEmbeddings(config['embeddings_path'])
    elif config['embeddings'] == 'bert':
        embeddings = BertEmbeddings()
    else:
        raise ValueError('Embeddings type not specified')

    embeddings = CachedEmbeddings(embeddings)

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
        channels_layer_1 = int(config.get('channels_layer_1', 100))
        channels_layer_2 = int(config.get('channels_layer_2', 100))

        kernel_size_layer_1 = int(config.get('kernel_size_layer_1', 3))
        kernel_size_layer_2 = int(config.get('kernel_size_layer_2', 3))

        activation = config.get('activation_function', 'relu')

        return CNN2d(embeddings=embeddings,
                     output_size=output_size,
                     sentence_length=int(config['sentence_length']),
                     channels_layer_1=channels_layer_1,
                     channels_layer_2=channels_layer_2,
                     kernel_size_layer_1=kernel_size_layer_1,
                     kernel_size_layer_2=kernel_size_layer_2,
                     activation_function=activation)

