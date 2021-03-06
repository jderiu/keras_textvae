# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import _pickle as cPickle
import getopt
import json
import logging
import os
import sys
from math import ceil
from os.path import join

import numpy as np
#do this before importing anything from Keras
np.random.seed(1337)
import keras.backend as K


from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, Nadam, Adadelta
from output_text import output_text
from sc_lstm_architecutre.sclstm_architecture_wordbased import vae_model
from data_loaders.data_loader_nlg import load_text_gen_data
from custom_callbacks import StepCallback, LexOutputCallback, TerminateOnNaN
import time


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

        tweets_path = config_data['tweets_path']
        vocab_path = config_data['vocab_path']
        vocab = cPickle.load(open(join(vocab_path, 'vocab_word.pkl'), 'rb'))

        #== == == == == == =
        # Load all the Data
        #== == == == == == =

        delimiter = ' '
        noutputs = 5

        logging.info('Load Training Data')
        train_input, train_output, train_lex = load_text_gen_data(join(tweets_path, 'trainset.csv'),   config_data, vocab, noutputs, word_based=True)
        logging.info('Load Validation Data')
        valid_input, valid_output, valid_lex = load_text_gen_data(join(tweets_path, 'devset.csv'), config_data, vocab, noutputs, word_based=True)
        logging.info('Load Output Validation Data')
        valid_dev_input, valid_dev_output, valid_dev_lex = load_text_gen_data(join(tweets_path, 'devset_reduced.csv'), config_data, vocab, noutputs, random_output=True, word_based=True)
        valid_dev_input2, valid_dev_output2, valid_dev_lex2 = load_text_gen_data(join(tweets_path, 'devset_reduced.csv'), config_data, vocab, noutputs, word_based=True)

        step = K.variable(1.)

        # == == == == == == == == == == =
        # Define and load the CNN model
        # == == == == == == == == == == =
        cnn_model, test_model = vae_model(config_data, vocab, step)
        cnn_model.save_weights(config_data['base_model_path'])
        cnn_model.summary()
        test_model.summary()

        model_path = 'models/vae_model/'
        steps_per_epoch = ceil(train_output[0].shape[0] / config_data['batch_size'])

        terminate_on_nan = TerminateOnNaN()
        model_checkpoint = ModelCheckpoint('models/vae_model/weights.{epoch:02d}.hdf5', period=10, save_weights_only=True)
        reduce_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.995, patience=100, min_lr=0.001, cooldown=50)

        #optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.001, clipnorm=10)
        optimizer = Adadelta(lr=1, epsilon=1e-8, rho=0.95, decay=0.0001)
        cnn_model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)

        vocab = {token: idx for token, (idx, freq) in vocab.items()}
        cnn_model.fit(
            x=train_input,
            y=train_output,
            epochs=1000,
            batch_size=config_data['batch_size'],
            validation_data=(valid_input, valid_output),
            callbacks=[StepCallback(step, steps_per_epoch),
                       LexOutputCallback(test_model, valid_dev_input, valid_dev_lex, 5, vocab, delimiter, fname='{}/test_output'.format(log_path)),
                       LexOutputCallback(test_model, valid_dev_input2, valid_dev_lex2, 5, vocab, delimiter, fname='{}/prec_test_output'.format(log_path)),
                       terminate_on_nan,
                       model_checkpoint,
                       reduce_callback],
        )

        cnn_out_path = join(config_data['output_path'], 'trained_deconv_vae_{}_model'.format(config_data['model_type']))
        cnn_model.save_weights(cnn_out_path)
        cnn_out_path = join(config_data['output_path'], 'trained_deconv_vae_{}_model_test'.format(config_data['model_type']))
        test_model.save_weights(cnn_out_path)

        output_text(test_model, valid_input, vocab)

if __name__ == '__main__':
    main(sys.argv[1:])
