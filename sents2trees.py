import os
import sys

from nltk.parse import corenlp
import pathlib

stanford_path = pathlib.Path('stanford/stanford-corenlp-full-2018-10-05')


with open(sys.argv[1]) as infile:
    sentences = [l.strip() for l in infile]

with corenlp.CoreNLPServer(
        path_to_jar=str(stanford_path / 'stanford-corenlp-3.9.2.jar'),
        path_to_models_jar=str(stanford_path / 'stanford-corenlp-3.9.2-models.jar')
        ) as server:

    parser = corenlp.CoreNLPParser()
    parsed_sentences = next(parser.raw_parse_sents(sentences))

for sentence in parsed_sentences:
    print(sentence)

