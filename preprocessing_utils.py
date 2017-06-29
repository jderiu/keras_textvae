# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np


def convert2indices(data, alphabet, dummy_word_idx, unk_word_idx, max_sent_length=140, verbose=0):
    data_idx = []
    max_len = 0
    unknown_words = 0
    known_words = 0
    for sentence in data:
        ex = np.ones(max_sent_length) * dummy_word_idx
        max_len = max(len(sentence), max_len)
        if len(sentence) > max_sent_length:
            sentence = sentence[:max_sent_length]
        for i, token in enumerate(sentence):
            idx = alphabet.get(token, unk_word_idx)
            ex[i] = idx
            if idx == unk_word_idx:
                unknown_words += 1
            else:
                known_words += 1
        data_idx.append(ex)
    data_idx = np.array(data_idx).astype('int32')
    if verbose == 1:
        print("Max length in this batch:", max_len)
        print("Number of unknown words:", unknown_words)
        print("Number of known words:", known_words)
    return data_idx