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
from keras.callbacks import Callback

from data_loader import load_data
from output_text import output_text

#do this before importing anything from Keras
np.random.seed(1337)
import keras.backend as K

class NewCallback(Callback):
    def __init__(self, alpha):
        self.alpha = alpha
    def on_epoch_end(self, epoch, logs={}):
        K.set_value(self.alpha, K.get_value(self.alpha) * epoch**0.95)


def main(args):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename='logging/evolution.log', filemode='w')
    try:
        opts, args = getopt.getopt(args, "c:", ["config="])
    except getopt.GetoptError:
        print('usage: -c config.json')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-c", "--config"):
            config_fname = os.path.join('configurations', arg)

    with open(config_fname, 'r') as json_data:
        config_data = json.load(json_data)


        #== == == == == == =
        # Load all the Data
        #== == == == == == =
        tweets_path = config_data['tweets_path']

        vocab = cPickle.load(open(join(tweets_path, 'vocabulary.pkl'), 'rb'))

        logging.info('Load Training Data')
        train_input = load_data(join(tweets_path, 'en_train.tsv'),   config_data, vocab)
        logging.info('Load Validation Data')
        valid_input= load_data(join(tweets_path, 'en_valid15.tsv'), config_data, vocab)
        logging.info('Load Validation Data')
        valid_input2 = load_data(join(tweets_path, 'en_test16.tsv'),  config_data, vocab)
        logging.info('Load Test Data')
        test_input = load_data(join(tweets_path, 'en_test17.tsv'),  config_data, vocab)

        step = K.variable(1.)

        if config_data['model_type'] == 'recurrent':
            from vae_architectures.vae_deconv_recurrent import vae_model
            train_net_input = [train_input, np.ones(len(train_input))*(2 + max(vocab.values()))]
            val_net_input = [valid_input, np.ones(len(valid_input))*(2 + max(vocab.values()))]
        else:
            from vae_architectures.vae_deconv_baseline import vae_model
            train_net_input = train_input
            val_net_input = valid_input

        # == == == == == == == == == == =
        # Define and load the CNN model
        # == == == == == == == == == == =
        cnn_model, test_model = vae_model(config_data, vocab, step)
        cnn_model.save_weights(config_data['base_model_path'])
        cnn_model.summary()


        cnn_model.fit(
            x=train_net_input,
            y=np.ones(len(train_input)),
            epochs=config_data['nb_epochs'],
            batch_size=config_data['batch_size'],
            shuffle=True,
            validation_data=(val_net_input, np.ones(len(valid_input))),
            callbacks=[NewCallback(step)]
        )

        output_text(test_model, valid_input, vocab)

if __name__ == '__main__':
    main(sys.argv[1:])
