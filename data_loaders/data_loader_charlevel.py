# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from os.path import join
from preprocessing_utils import convert2indices
import _pickle as cPickle
from collections import defaultdict
import gzip
import numpy as np
import random
import csv


def generate_data_stream(fname, config_data, vocabulary, batch_size, noutputs=2, skip_data=0):
    max_sentence_len = config_data['max_sentence_len']
    max_idx = max(vocabulary.values())
    dummy_word_idx = max_idx + 1
    outputs = [np.ones(batch_size)] * noutputs

    current_batch = []
    while True:
        if fname.endswith('.tsv') or fname.endswith('.txt'):
            ifile = open(fname, mode='rt', encoding='utf-8')
        elif fname.endswith('.gz') or fname.endswith('.gzip'):
            ifile = gzip.open(fname, mode='rt', encoding='utf-8')

        for i, line in enumerate(ifile, start=1):
            if not skip_data == 0:
                skip_data -= 1
                continue

            current_batch.append(line)
            if i % batch_size == 0:
                random.shuffle(current_batch)
                processed_batch = [x.replace('\r', '').split('\t')[-1] for x in current_batch]
                batch_idx = convert2indices(processed_batch, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_sentence_len)
                yield [batch_idx, batch_idx], outputs
                current_batch = []
        ifile.close()


def load_text_pairs(fname, config_data, vocabulary, noutputs=3):
    max_input_length = config_data['max_input_length']
    max_output_length = config_data['max_output_length']
    max_idx = max(vocabulary.values())
    dummy_word_idx = max_idx + 1

    ifile = open(fname, encoding='utf-8', mode='rt')
    inputs_raw = []
    outputs_raw = []
    for line in ifile:
        sline = line.replace('\n', '').split('\t')
        text0 = sline[0]
        text1 = sline[1]

        inputs_raw.append(text0)
        outputs_raw.append(text1)

    input_idx = convert2indices(inputs_raw, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_input_length)
    target_idx = convert2indices(outputs_raw, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_output_length)

    outputs = [np.ones(len(input_idx))] * noutputs
    return [input_idx, target_idx], outputs


def load_text_gen_data(fname, config_data, vocabulary, noutputs=3):
    max_input_length = config_data['max_input_length']
    max_output_length = config_data['max_output_length']
    max_idx = max(vocabulary.values())
    dummy_word_idx = max_idx + 1
    reader = csv.DictReader(open(fname, encoding='utf-8', mode='rt'))

    inputs_raw = []
    outputs_raw = []
    for row in reader:
        i1 = row['mr']
        i2 = row['ref']

        inputs_raw.append(i1)
        outputs_raw.append(i2)

    input_idx = convert2indices(inputs_raw, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_input_length)
    target_idx = convert2indices(outputs_raw, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_output_length)

    outputs = [np.ones(len(input_idx))] * noutputs

    return [input_idx, target_idx], outputs


def load_data(fname, config_data, vocabulary, noutputs=2):
    max_sentence_len = config_data['max_sentence_length']

    input_data, output_data = transform_data(fname, vocabulary, max_sentence_len, noutputs)

    return input_data, output_data


def transform_data(fname, vocabulary, max_sentence_len, noutputs):
    max_idx = max(vocabulary.values())
    dummy_word_idx = max_idx + 1

    file = open(fname, encoding='utf-8', mode='rt')

    curr_tweets = [x.replace('\r', '').split('\t')[-1] for x in file.readlines()]
    text_idx = convert2indices(curr_tweets, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_sentence_len)
    outputs = [np.ones(len(curr_tweets))] * noutputs

    return [text_idx, text_idx], outputs


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
        if f > 0:
            vocab_idx[v] = counter
            counter += 1
    return vocab_idx

if __name__ == "__main__":
    dir = 'F:/traindev'
    files = [
        join(dir, 'devel-conc.txt'),
        join(dir, 'train-conc.txt')
    ]

    vocab_idx = create_vocabulary(files)
    cPickle.dump(vocab_idx, open('F:/traindev/vocabulary.pkl', 'wb'))
    print(len(vocab_idx))