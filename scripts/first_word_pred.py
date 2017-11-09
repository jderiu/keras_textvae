# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import _pickle as cPickle
import getopt
import json
import logging
import os
import sys
from os.path import join

import numpy as np
from data_loaders.data_loader_nlg import load_text_gen_data
import time
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

def main(args):
    try:
        opts, args = getopt.getopt(args, "c:s", ["config="])
    except getopt.GetoptError:
        print('usage: -c config.json')
        sys.exit(2)

    start_from_model = False
    for opt, arg in opts:
        if opt in ("-c", "--config"):
            config_fname = os.path.join('configurations', arg)
        elif opt == '-s':
            start_from_model = True

    if start_from_model:
        filemode = 'a'
    else:
        filemode = 'w'

    log_path = 'logging/vae_nlg_{}'.format(int(round(time.time() * 1000)))
    os.mkdir(log_path)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename='{}/evolution.log'.format(log_path), filemode=filemode)

    with open(config_fname, 'r') as json_data:
        config_data = json.load(json_data)

        batch_size = config_data['batch_size']
        epochs = config_data['nb_epochs']
        discriminator_iterations = config_data['discriminator_iterations']
        tweets_path = config_data['tweets_path']
        vocab_path = config_data['vocab_path']
        vocab = cPickle.load(open(join(vocab_path, 'vocabulary.pkl'), 'rb'))
        vocab_words = cPickle.load(open(join(vocab_path, 'fw_vocab.pkl'), 'rb'))
        inv_fw_vocab = {v: k for k,v in vocab_words.items()}

        model_path = config_data['output_path']
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # == == == == == == =
        # Load all the Data
        # == == == == == == =
        delimiter = ''
        noutputs = 10

        logging.info('Load Training Data')
        train_input, train_output, train_weights, train_lex = load_text_gen_data(join(tweets_path, 'trainset.csv'), config_data, vocab, noutputs, word_based=False)
        logging.info('Load Validation Data')
        valid_input, valid_output, _, valid_lex = load_text_gen_data(join(tweets_path, 'devset.csv'), config_data,vocab, noutputs, word_based=False)
        logging.info('Load Output Validation Data')
        valid_dev_input, valid_dev_output, _, valid_dev_lex = load_text_gen_data(join(tweets_path, 'devset_reduced.csv'), config_data, vocab, noutputs, random_output=True, word_based=False)
        valid_dev_input2, valid_dev_output2, _, valid_dev_lex2 = load_text_gen_data(join(tweets_path, 'devset_located.csv'), config_data, vocab, noutputs, random_output=True, word_based=False,random_first_word=True)
        valid_dev_input3, valid_dev_output3, _, valid_dev_lex3 = load_text_gen_data(join(tweets_path, 'test_e2e.csv'),config_data, vocab, noutputs,random_output=True,word_based=False,random_first_word=True)

        x_train = np.asarray([np.concatenate(x) for x in zip(*train_input[:8])])
        y_train = train_input[8]

        x_dev = np.asarray([np.concatenate(x) for x in zip(*valid_input[:8])])
        y_dev = valid_input[8]

        x_test = np.asarray([np.concatenate(x) for x in zip(*valid_input[:8])])
        y_test = valid_input[8]

        print(x_train.shape)
        print(y_train.shape)

        #y_count = np.sum(y_train, axis=0)
        #y_cw = y_train.shape[0]/(y_train.shape[1]*y_count)
        #class_weights = {i: cw for i, cw in enumerate(y_cw)}
        model = SGDClassifier(loss='squared_hinge', penalty='l1',verbose=1, class_weight='balanced')
        model.fit(x_train, np.argmax(y_train, axis=1))

        y_pred = model.predict(x_dev)

        print(accuracy_score(np.argmax(y_dev, axis=1), y_pred))
        print(f1_score(np.argmax(y_dev, axis=1),y_pred, average=None))

        y_pred = model.predict(x_test)
        for yp in y_pred:
            value = inv_fw_vocab.get(yp, 'NONE')
            print(value)


if __name__ == '__main__':
    main(sys.argv[1:])

