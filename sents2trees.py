import os
import sys

from nltk.parse import corenlp
import pathlib

stanford_path = pathlib.Path('stanford/stanford-corenlp-full-2018-10-05')
url='http://localhost:9001'

sentences = []
with open(sys.argv[1]) as infile:
    for line in infile:
        sentences.extend([s.strip() for s in line.split('\t')])


parser = corenlp.CoreNLPParser(url=url)
for sentence in sentences:
    parsed_sentence = parser.raw_parse(sentence)
    print(next(parsed_sentence))
    print()

