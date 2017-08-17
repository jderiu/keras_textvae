"""
This Dataloader is an iterator for long files, mostly used for sentiment&language detection task.
If multiple files are used they are processed sequencially. Look at the mixed_iterator if the file content needs to be mixed.
This is expecially useful if the Task is based only on a single large file.
"""

from __future__ import unicode_literals

import numpy as np
from preprocessing_utils import hybrid_convert2indices, preprocess, convert2indices
import gzip
import random
import json
from os.path import join
import _pickle as cPickle


def generate_data_stream(fname, config_data, vocabulary_char, vocabulary_word, batch_size, noutputs=2, skip_data=0):
    max_sentence_len = config_data['max_sentence_length']
    dummy_word_idx = vocabulary_word['DUMMY_WORD'][0]
    dummy_char_idx = max(vocabulary_char.values()) + 1
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

            current_batch.append(line.split('\t')[-1])
            if i % batch_size == 0:
                random.shuffle(current_batch)
                processed_batch = [preprocess(x.replace('\r', '')) for x in current_batch]
                batch_word_idx = hybrid_convert2indices(current_batch, processed_batch, vocabulary_word, dummy_word_idx, dummy_word_idx, max_sent_length=max_sentence_len)
                batch_char_idx = convert2indices(current_batch, vocabulary_char, dummy_char_idx, dummy_char_idx, max_sent_length=max_sentence_len)

                yield [batch_char_idx, batch_word_idx], outputs
                current_batch = []
        ifile.close()


def load_data(fname, config_data, vocabulary_word, vocabulary_char, noutputs):
    max_sentence_len = config_data['max_sentence_length']

    input_data, output_data = transform_data(fname, vocabulary_word, vocabulary_char, max_sentence_len, noutputs)

    return input_data, output_data


def transform_data(fname, vocabulary_word, vocabulary_char, max_sentence_len, noutputs):
    dummy_word_idx = vocabulary_word['DUMMY_WORD'][0]
    dummy_char_idx = max(vocabulary_char.values()) + 1

    file = open(fname, encoding='utf-8', mode='rt')

    curr_tweets = [x.replace('\r', '').split('\t')[-1] for x in file.readlines()]
    processed_batch = [preprocess(x) for x in curr_tweets]
    text_idx = hybrid_convert2indices(curr_tweets, processed_batch, vocabulary_word, dummy_word_idx, dummy_word_idx, max_sent_length=max_sentence_len)
    char_idx = convert2indices(curr_tweets, vocabulary_char, dummy_char_idx, dummy_char_idx,max_sent_length=max_sentence_len)
    outputs = [np.ones(len(curr_tweets))] * noutputs

    return [char_idx, text_idx], outputs

if __name__ == "__main__":
    config_data = json.load(open('configurations/config_vae_hybrid.json'))
    vocab_word_path = join(config_data['vocab_word_path'], 'vocabulary.pkl')
    vocab_char_path = join(config_data['vocab_char_path'], 'vocabulary.pkl')

    vocab_word = cPickle.load(open(vocab_word_path, 'rb'))
    vocab_char = cPickle.load(open(vocab_char_path, 'rb'))
    for b, o in generate_data_stream(config_data['training_path'], config_data, vocab_char, vocab_word, config_data['batch_size'], skip_data=0, noutputs=2):
        pass


