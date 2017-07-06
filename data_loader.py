# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from os.path import join
from preprocessing_utils import convert2indices
import _pickle as cPickle
from collections import defaultdict
import gzip
import numpy as np

senti_map = {
    'negative': 0,
    'positive': 2,
    'neutral': 1
}


def generate_data_stream(fname, config_data, vocabulary, batch_size):
    max_sentence_len = config_data['max_sentence_length']
    max_idx = max(vocabulary.values())
    dummy_word_idx = max_idx + 1

    while True:
        if fname.endswith('.tsv'):
            ifile = open(fname, mode='rt', encoding='utf-8')
        elif fname.endswith('.gz') or fname.endswith('.gzip'):
            ifile = gzip.open(fname, mode='rt', encoding='utf-8')
        current_batch = []
        for i, line in enumerate(ifile):
            current_batch.append(line)
            if i % batch_size == 0:
                processed_batch = [x.replace('\r', '').split('\t')[-1] for x in current_batch]
                batch_idx = convert2indices(processed_batch, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_sentence_len)
                yield (batch_idx, np.ones(len(batch_idx)))
                current_batch = []
        ifile.close()


def load_data(fname, config_data, vocabulary):
    max_sentence_len = config_data['max_sentence_length']

    input_data = transform_data(fname, vocabulary, max_sentence_len)

    return input_data


def transform_data(fname, vocabulary, max_sentence_len):
    max_idx = max(vocabulary.values())
    dummy_word_idx = max_idx + 1
    start_word = max_idx + 2

    file = open(fname, encoding='utf-8', mode='rt')

    curr_tweets = [x.replace('\r', '').split('\t')[-1] for x in file.readlines()]
    text_idx = convert2indices(curr_tweets, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_sentence_len)

    return text_idx


def create_vocabulary(files):
    vocab_freq = defaultdict(lambda: 0)
    for fname in files:
        idata = open(fname, 'rt', encoding='utf-8', newline='\n').readlines()
        for sample in idata:
            text = sample.split('\t')[-1]
            for c in text:
                vocab_freq[c] += 1

    vocab_idx = {}
    counter = 0
    for i, (v, f) in enumerate(vocab_freq.items()):
        if f > 10:
            vocab_idx[v] = counter
            counter += 1
    return vocab_idx

if __name__ == "__main__":
    dir = 'en_full'
    files = [
        join(dir, 'en_train.tsv'),
        join(dir, 'en_test16.tsv'),
        join(dir, 'en_test17.tsv'),
        join(dir, 'en_valid15.tsv')
    ]

    vocab_idx = create_vocabulary(files)
    cPickle.dump(vocab_idx, open('en_full/vocabulary.pkl', 'wb'))