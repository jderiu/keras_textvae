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
import time

from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, Nadam, Adadelta
from output_text import output_text
from custom_callbacks import TerminateOnNaN, OutputCallback, StepCallback


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

    curr_time = time.time()
    log_path = 'logging/vae_nlg_{}'.format(int(round(curr_time * 1000)))
    os.mkdir(log_path)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename='{}/evolution.log'.format(log_path), filemode=filemode)

    with open(config_fname, 'r') as json_data:
        config_data = json.load(json_data)

        tweets_path = config_data['tweets_path']
        vocab_path = config_data['vocab_path']
        vocab = cPickle.load(open(join(vocab_path, 'vocabulary.pkl'), 'rb'))

        #== == == == == == =
        # Load all the Data
        #== == == == == == =
        if config_data['model_type'] == 'recurrent':
            from vae_architectures.vae_deconv_recurrent import vae_model
            from data_loaders.data_loader_charlevel import load_data, generate_data_stream
            delimiter = ''
            noutputs = 3
        elif config_data['model_type'] == 'baseline_word':
            from vae_architectures.vae_deconv_baseline_word import vae_model
            from data_loaders.data_loader_wordlevel import load_data, generate_data_stream
            vocab = {k: v[0] for k, v in vocab.items()}
            noutputs = 2
            delimiter = ' '
        elif config_data['model_type'] == 'recurrent_word':
            from vae_architectures.vae_deconv_recurrent_word import vae_model
            from data_loaders.data_loader_wordlevel import load_data, generate_data_stream
            vocab = {k: v[0] for k, v in vocab.items()}
            noutputs = 3
            delimiter = ' '
        else:
            from vae_architectures.vae_deconv_baseline import vae_model
            from data_loaders.data_loader_charlevel import load_data, generate_data_stream
            noutputs = 1
            delimiter = ''

        logging.info('Load Training Data')
        train_input, train_output = load_data(join(tweets_path, 'en_train.tsv'),   config_data, vocab, noutputs)
        logging.info('Load Validation Data')
        valid_input, valid_output = load_data(join(tweets_path, 'en_valid15.tsv'), config_data, vocab, noutputs)

        step = K.variable(1.)

        model_path = 'models/vae_model_{}'.format(curr_time)
        os.mkdir(model_path)
        # == == == == == == == == == == =
        # Define and load the CNN model
        # == == == == == == == == == == =
        cnn_model, test_model = vae_model(config_data, vocab, step)
        cnn_model.save_weights(os.path.join(model_path, 'base_cnn_model.h5'))
        cnn_model.summary()



        steps_per_epoch = ceil(config_data['samples_per_epoch'] / config_data['batch_size'])
        initial_epoch = 0

        terminate_on_nan = TerminateOnNaN()
        model_checkpoint = ModelCheckpoint(os.path.join(model_path, 'weights.{epoch:02d}.hdf5'), period=10, save_weights_only=True)
        reduce_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.995, patience=10, min_lr=0.001, cooldown=50)

        #optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.001, clipnorm=10)
        optimizer = Adadelta(lr=1.0, epsilon=1e-8, rho=0.95, decay=0.0001, clipnorm=10)
        cnn_model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)
        cnn_model.fit_generator(
            generator=generate_data_stream(config_data['training_path'], config_data, vocab, config_data['batch_size'], noutputs=noutputs),
            steps_per_epoch=steps_per_epoch,
            epochs=ceil(config_data['nb_epochs']*(config_data['nsamples']/config_data['samples_per_epoch'])),
            callbacks=[StepCallback(step, steps_per_epoch), OutputCallback(test_model, valid_input[0], 5, vocab, delimiter, fname='{}/test_output'.format(log_path)), terminate_on_nan, model_checkpoint, reduce_callback],
            validation_data=(valid_input, valid_output),
            initial_epoch=initial_epoch
        )

        test_model.summary()

        cnn_out_path = join(config_data['output_path'], 'trained_deconv_vae_{}_model'.format(config_data['model_type']))
        cnn_model.save_weights(cnn_out_path)
        cnn_out_path = join(config_data['output_path'], 'trained_deconv_vae_{}_model_test'.format(config_data['model_type']))
        test_model.save_weights(cnn_out_path)

        output_text(test_model, valid_input, vocab)

if __name__ == '__main__':
    main(sys.argv[1:])
