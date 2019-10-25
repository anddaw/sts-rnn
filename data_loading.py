from pathlib import Path
from typing import Tuple, List

import torch

from tree import TreeReader

from data_preprocessor import DataPreprocessor
from utils import log


def _get_label(s: float, approximate_distribution: bool = False, labels_size: int = 6):
    """Convert float to labels probabilities vector"""

    t = torch.zeros(labels_size)
    if approximate_distribution:
        if s >= labels_size - 1:
            t[-1] = 1
        else:
            t[int(s) + 1] = s - int(s)
            t[int(s)] = 1 - t[int(s) + 1]
    else:
        t[int(round(s))] = 1
    return t


def load_labels(path: str, rounding: str = 'int') -> Tuple[List[torch.Tensor], List[float]]:
    """Load scores and labels from file"""

    if rounding == 'int':
        approx = False
    elif rounding == 'approx':
        approx = True
    else:
        raise ValueError('Unknown rounding type')

    with open(path, 'r') as labels_file:
        scores = [float(s.strip()) for s in labels_file]

    labels = [_get_label(s, approximate_distribution=approx) for s in scores]

    return labels, scores


def load_sentence_pairs(path: str) -> List[str]:

    with open(path, 'r') as sentences_file:
        return list(sentences_file)


def load_dataset(dataset, config):

    if dataset not in ['train', 'test']:
        raise ValueError(f'Unknown dataset {dataset}')

    log(f'Loading {dataset} data')

    input_type = config['input_type']
    dataset_path = Path(config['dataset'])
    labels_path = dataset_path / f'{dataset}_labels.txt'
    tree_suffix = '_tree' if config['input_type'] == 'tree' else ''
    sentences_path = dataset_path / f'{dataset}{tree_suffix}.txt'

    if input_type == 'sentence':
        preprocessor = DataPreprocessor()
        sentences = preprocessor.preprocess_sentence_pairs(load_sentence_pairs(sentences_path))
    elif input_type == 'tree':
        reader = TreeReader(DataPreprocessor())
        sentences = reader.load_from_file(sentences_path)
    else:
        raise ValueError(f'Unknown input_type: {input_type}')

    labels, similarities = load_labels(labels_path, rounding=config['label_rounding'])
    return sentences, labels, similarities


