# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import getopt
import json
import logging
from os.path import join
import sys

import argparse
import _pickle as cPickle
import numpy as np
from gensim.models import Word2Vec
from data_loaders.data_loader_nlg import get_texts
from collections import Counter

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
config_data = json.load(open('configurations/config_vae_nlg.json'))
tweets_path = config_data['tweets_path']

train_texts = get_texts(join(tweets_path, 'trainset.csv'))
dev_texts = get_texts(join(tweets_path, 'devset.csv'))

full_tokens = Counter()
sentence_lengths = []
for tokens in train_texts + dev_texts:
    full_tokens.update(tokens)
    sentence_lengths.append(len(tokens))

print(len(full_tokens))
print(full_tokens.most_common(100))
sentence_lengths = np.array(sentence_lengths)
print(np.mean(sentence_lengths))
print(np.std(sentence_lengths))

vocabulary = {}
for i, (token, freq) in enumerate(full_tokens.items()):
    vocabulary[token] = i

cPickle.dump(vocabulary, open(join(tweets_path, 'vocab_word.pkl'), 'wb'))

model = Word2Vec(
    train_texts + dev_texts,
    size=200,
    window=4,
    min_count=0,
    workers=2,
    sg=1,
    sample=1e-5,
    hs=1,
    iter=20,
    negative=10,
    max_vocab_size=5000
)

model.wv.save_word2vec_format(join(tweets_path, 'embedding_file'), binary=False)

vocab_emb = np.zeros((len(vocabulary) + 1, 200), dtype='float32')
for word, idx in vocabulary.items():
    word_vec = None
    if word in model.wv.vocab.keys():
        word_vec = model[word]
    if word_vec is None:
        word_vec = np.random.uniform(-0.25, 0.25, 200)
    vocab_emb[idx] = word_vec

outfile = join(tweets_path, 'embedding_matrix.npy')