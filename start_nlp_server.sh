#!/usr/bin/bash

/usr/bin/java -mx2g -cp stanford/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar:stanford/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2-models.jar edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,parse,depparse -port 9001
