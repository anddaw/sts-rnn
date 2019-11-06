from argparse import ArgumentParser

import torch

from models.base import BaseModel
from data_preprocessor import DataPreprocessor


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('sentences')
    parser.add_argument('model')

    args = parser.parse_args()

    model: BaseModel = torch.load(args.model)
    model.eval()

    with open(args.sentences) as fh:
        sentences = list(fh)

    sentences = DataPreprocessor().preprocess_sentence_pairs(sentences)

    for sentence in sentences:
        print(model.predict_score(sentence))



