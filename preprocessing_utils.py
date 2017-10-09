# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
from nltk.tokenize import TweetTokenizer, WordPunctTokenizer
import re


def preprocess_nlg_text(text, name, near, word_based=True):
    name_tok = 'XNAME'
    near_tok = 'XNEAR'

    text = text.replace('\n', '').replace('\r', '').replace('\t', ' ')

    if name is not '':
        text = text.replace(name, name_tok)

    if near is not '':
        text = text.replace(near, near_tok)

    if word_based:
        tokenizer = WordPunctTokenizer()
        tokens = tokenizer.tokenize(text)
    else:
        tokens = text

    return tokens

def preprocess(tweet):
    tokenzer = TweetTokenizer()
    #lowercase and normalize urls
    tweet = tweet.lower()
    tweet = tweet.replace('\n', '')
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', 'URLTOK', tweet)
    tweet = re.sub('@[^\s]+', 'USRTOK', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    tweet = tokenzer.tokenize(tweet)

    return list(map(lambda x: x.replace(' ', ''), tweet))


def preprocess_char_x_word(tweet):
    tokenzer = TweetTokenizer()
    # lowercase and normalize urls
    tweet = tweet.lower()
    tweet = tweet.replace('\n', '')
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', 'URLTOK', tweet)
    tweet = re.sub('@[^\s]+', 'USRTOK', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    tweet_tok = tokenzer.tokenize(tweet)

    return list(map(lambda x: x.replace(' ', ''), tweet_tok))


def convert2indices(data, alphabet, dummy_word_idx, unk_word_idx, max_sent_length=140, verbose=0):
    data_idx = []
    max_len = 0
    unknown_words = 0
    known_words = 0
    for sentence in data:
        ex = np.ones(max_sent_length)*dummy_word_idx
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
    data_idx = np.array(data_idx).astype('float32')
    if verbose == 1:
        print("Max length in this batch:", max_len)
        print("Number of unknown words:", unknown_words)
        print("Number of known words:", known_words)
    return data_idx


def hybrid_convert2indices(data, tokenized_data, alphabet, dummy_word_idx, unk_word_idx, max_sent_length=128, verbose=0):
    data_idx = []
    max_len = 0
    unknown_words = 0
    known_words = 0
    for sentence, sent_toks in zip(data, tokenized_data):
        sentence = sentence.lower()
        ex = np.ones(max_sent_length)*dummy_word_idx
        max_len = max(len(sent_toks), max_len)
        if len(sent_toks) > max_sent_length:
            sent_toks = sent_toks[:max_sent_length]
        sent_ptr = 0
        for i, token in enumerate(sent_toks):
            vocab_idx = alphabet.get(token, (unk_word_idx, 1))[0]
            sidx = sentence.find(token, sent_ptr)
            ex[sidx:sidx + len(token)] = vocab_idx
            if vocab_idx == unk_word_idx:
                unknown_words += 1
            else:
                known_words += 1
        data_idx.append(ex)
    data_idx = np.array(data_idx).astype('float32')
    if verbose == 1:
        print("Max length in this batch:", max_len)
        print("Number of unknown words:", unknown_words)
        print("Number of known words:", known_words)
    return data_idx
