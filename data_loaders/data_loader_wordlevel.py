"""
This Dataloader is an iterator for long files, mostly used for sentiment&language detection task.
If multiple files are used they are processed sequencially. Look at the mixed_iterator if the file content needs to be mixed.
This is expecially useful if the Task is based only on a single large file.
"""

from __future__ import unicode_literals

import numpy as np
from preprocessing_utils import convert2indices, preprocess
import gzip
import random
import json
from os.path import join
import _pickle as cPickle


def generate_data_stream(fname, config_data, vocabulary, batch_size, noutputs=2, skip_data=0):
    max_sentence_len = config_data['max_sentence_length']
    dummy_word_idx = vocabulary['DUMMY_WORD']
    outputs = [np.ones(batch_size)]*noutputs
    #vocabulary = {k: v[0] for k, v in vocabulary.items()}
    current_batch = []
    while True:
        if fname.endswith('.tsv'):
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
                processed_batch = [preprocess(x.replace('\r', '').split('\t')[-1]) for x in current_batch]
                batch_idx = convert2indices(processed_batch, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_sentence_len)
                yield batch_idx, outputs
                current_batch = []
        ifile.close()


def load_data(fname, config_data, vocabulary, noutputs):
    max_sentence_len = config_data['max_sentence_length']

    input_data, output_data = transform_data(fname, vocabulary, max_sentence_len, noutputs)

    return input_data, output_data


def transform_data(fname, vocabulary, max_sentence_len, noutputs):
    dummy_word_idx = vocabulary['DUMMY_WORD']

    file = open(fname, encoding='utf-8', mode='rt')

    curr_tweets = [x.replace('\r', '').split('\t')[-1] for x in file.readlines()]
    processed_batch = [preprocess(x) for x in curr_tweets]
    text_idx = convert2indices(processed_batch, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_sentence_len)
    outputs = [np.ones(len(curr_tweets))] * noutputs

    return text_idx, outputs

if __name__ == "__main__":
    config_data = json.load(open('../configurations/config_vae_word.json'))
    vocab_path = join(config_data['vocab_path'], 'vocabulary.pkl')

    vocab = cPickle.load(open(vocab_path, 'rb'))
    word_sum = sum([v[1] for k, v in vocab.items()])
    print(word_sum)
    idx_to_rel_freq = {v[0]: v[1]/word_sum for k, v in vocab.items()}
    sorted_idx_to_freq = sorted(idx_to_rel_freq.items(), key=lambda v: v[1], reverse=True)
    print(sorted_idx_to_freq[:10])
    freq_vec = np.ones(shape=(len(sorted_idx_to_freq), 1))

    for k, v in sorted_idx_to_freq:
        freq_vec[k] = v

    e_x = np.exp(freq_vec - np.max(freq_vec))
    softmax_freq = e_x/e_x.sum()
    print(softmax_freq[-1])

