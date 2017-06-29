# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import json
import logging
import sys
import os
import getopt
import _pickle as cPickle
import numpy as np
from data_loader import load_data
from os.path import join
#do this before importing anything from Keras
np.random.seed(1337)

from vae import vae_model
import keras.backend as K



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
        tweets_vec_path = config_data['tweets_vec_path']

        vocab = cPickle.load(open('en_full/vocabulary.pkl', 'rb'))

        logging.info('Load Training Data')
        train_input = load_data(join(tweets_path, 'en_train.tsv'),   config_data, vocab)
        logging.info('Load Validation Data')
        valid_input= load_data(join(tweets_path, 'en_valid15.tsv'), config_data, vocab)
        logging.info('Load Validation Data')
        valid_input2 = load_data(join(tweets_path, 'en_test16.tsv'),  config_data, vocab)
        logging.info('Load Test Data')
        test_input = load_data(join(tweets_path, 'en_test17.tsv'),  config_data, vocab)


        # == == == == == == == == == == =
        # Define and load the CNN model
        # == == == == == == == == == == =
        cnn_model = vae_model(config_data, vocab)
        cnn_model.save_weights(config_data['base_model_path'])
        cnn_model.summary()

        cnn_model.fit(
            x=train_input,
            y=np.ones(len(train_input)),
            epochs=10,
            shuffle=True,
            validation_data=(valid_input, np.ones(len(valid_input)))
        )

if __name__ == '__main__':
    main(sys.argv[1:])
