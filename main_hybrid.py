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


class NewCallback(Callback):
    def __init__(self, alpha, steps_per_epoch, **kwargs):
        self.alpha = alpha
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0
        super(NewCallback, self).__init__(**kwargs)

    def on_batch_end(self, batch, logs=None):
        value = self.steps_per_epoch*self.current_epoch + batch
        K.set_value(self.alpha, value)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch


class OutputCallback(Callback):
    def __init__(self, test_model, validation_input, frequency, vocabulary, delimiter, **kwargs):
        self.validation_input = validation_input
        self.vocabulary = vocabulary
        self.test_model = test_model
        self.frequency = frequency
        self.delimiter = delimiter
        super(OutputCallback, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == 0:
            output_text(self.test_model, self.validation_input, self.vocabulary, str(epoch), delimiter=self.delimiter)

        main_loss = logs.get('main_loss_loss', '-')
        kld_loss = logs.get('kld_loss_char_loss', '-')
        kld_loss_word = logs.get('kld_loss_word_loss', '-')
        auxiliary_loss = logs.get('auxiliary_loss_loss', '-')
        auxiliary_word_loss = logs.get('auxiliary_word_loss_loss', '-')
        val_main_loss = logs.get('val_main_loss_loss', '-')
        val_kld_loss = logs.get('val_kld_loss_char_loss', '-')
        val_kld_loss_word = logs.get('val_kld_loss_word_loss', '-')
        val_auxiliary_loss = logs.get('val_auxiliary_loss_loss', '-')
        val_auxiliary_word_loss = logs.get('val_auxiliary_word_loss_loss', '-')

        logging.info('Epoch: {}'.format(epoch))
        logging.info('TRA: Total Loss: {}\t Main Loss: {}\tKLD Char Loss: {}\tKLD Word Loss: {}\tAux Char Loss: {}\t Aux Word Loss: {}'.
                     format(logs['loss'], main_loss, kld_loss, kld_loss_word, auxiliary_loss, auxiliary_word_loss))
        logging.info('VAL: Total Loss: {}\t Main Loss: {}\tKLD Loss: {}\tKLD Word Loss: {}\tAux Char Loss: {}\t Aux Word Loss: {}'.
                     format(logs['val_loss'], val_main_loss, val_kld_loss, val_kld_loss_word, val_auxiliary_loss,val_auxiliary_word_loss))
        #reset datastructures
        self.ep_begin_weights = {}
        self.ep_end_weights = {}


class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered."""

    def __init__(self):
        self.terminated_on_nan = False
        super(TerminateOnNaN, self).__init__()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True
                self.terminated_on_nan = True


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

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename='logging/evolution.log', filemode=filemode)

    with open(config_fname, 'r') as json_data:
        config_data = json.load(json_data)

        tweets_path = config_data['tweets_path']
        vocab_word_path = config_data['vocab_word_path']
        vocab_char_path = config_data['vocab_char_path']

        vocab_word = cPickle.load(open(join(vocab_word_path, 'vocabulary.pkl'), 'rb'))
        vocab_char = cPickle.load(open(join(vocab_char_path, 'vocabulary.pkl'), 'rb'))

        #== == == == == == =
        # Load all the Data
        #== == == == == == =
        from vae_architectures.siamese_vae import vae_model
        from data_loaders.data_loader_hybrid import load_data, generate_data_stream
        delimiter = ''
        noutputs = 5

        logging.info('Load Training Data')
        train_input, train_output = load_data(join(tweets_path, 'en_train.tsv'),   config_data, vocab_word, vocab_char, noutputs)
        logging.info('Load Validation Data')
        valid_input, valid_output = load_data(join(tweets_path, 'en_valid15.tsv'), config_data, vocab_word, vocab_char, noutputs)
        logging.info('Load Validation Data')
        valid_input2, valid_output2 = load_data(join(tweets_path, 'en_test16.tsv'),  config_data, vocab_word, vocab_char, noutputs)
        logging.info('Load Test Data')
        test_input, test_output = load_data(join(tweets_path, 'en_test17.tsv'),  config_data, vocab_word, vocab_char, noutputs)

        step = K.variable(1.)

        # == == == == == == == == == == =
        # Define and load the CNN model
        # == == == == == == == == == == =
        cnn_model, test_model = vae_model(config_data, vocab_char, step)
        cnn_model.save_weights(config_data['base_model_path'])
        cnn_model.summary()

        model_path = 'models/vae_model/'
        steps_per_epoch = ceil(config_data['samples_per_epoch'] / config_data['batch_size'])

        delta_epoch = 0
        last_initial_epoch = 0
        initial_epoch = 0
        delta_increase = 1
        loaded_epoch = -1
        number_of_increases = 0
        while True:
            skip_texts = 0
            if start_from_model:
                file_names = [(x, int(x.split('.')[1])) for x in os.listdir(model_path)]

                latest_fname, loaded_epoch = sorted(file_names, key=lambda x: x[1], reverse=True)[0]
                if loaded_epoch > last_initial_epoch:
                    delta_epoch = 0
                    number_of_increases = 0

                #sign that the current model is broken
                if number_of_increases > 3:
                    latest_fname, loaded_epoch = sorted(file_names, key=lambda x: x[1], reverse=True)[1]

                initial_epoch = loaded_epoch + delta_epoch
                fname = os.path.join(model_path, latest_fname)
                cnn_model.load_weights(fname)
                skip_texts = initial_epoch*config_data['samples_per_epoch']
                logging.info('Resume Training from Epoch: {}. Skipping {} Datapoints'.format(initial_epoch, skip_texts))

            terminate_on_nan = TerminateOnNaN()
            model_checkpoint = ModelCheckpoint('models/vae_model/weights.{epoch:02d}.hdf5', period=10, save_weights_only=True)
            reduce_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.995, patience=100, min_lr=0.001, cooldown=50)

            optimizer = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.001, clipnorm=10)
            #optimizer = Adadelta(lr=1.0, epsilon=1e-8, rho=0.95, decay=0.0001, clipnorm=10)
            cnn_model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)
            cnn_model.fit_generator(
                generator=generate_data_stream(config_data['training_path'], config_data, vocab_char, vocab_word, config_data['batch_size'], skip_data=skip_texts, noutputs=noutputs),
                steps_per_epoch=steps_per_epoch,
                epochs=ceil(config_data['nb_epochs']*(config_data['nsamples']/config_data['samples_per_epoch'])),
                callbacks=[NewCallback(step, steps_per_epoch), OutputCallback(test_model, valid_input, 15, vocab_char, delimiter), terminate_on_nan, model_checkpoint, reduce_callback],
                validation_data=(valid_input, valid_output),
                initial_epoch=initial_epoch
            )
            if terminate_on_nan.terminated_on_nan:
                start_from_model = True
                #if failed twice in same epoch -> skip a couple of epochs
                if last_initial_epoch >= loaded_epoch:
                    delta_epoch += delta_increase
                    number_of_increases += 1
                    logging.info(msg='Skip 1 epoch')

                last_initial_epoch = initial_epoch
                terminate_on_nan.terminated_on_nan = False
            else:
                break

        test_model.summary()

        cnn_out_path = join(config_data['output_path'], 'trained_deconv_vae_{}_model'.format(config_data['model_type']))
        cnn_model.save_weights(cnn_out_path)
        cnn_out_path = join(config_data['output_path'], 'trained_deconv_vae_{}_model_test'.format(config_data['model_type']))
        test_model.save_weights(cnn_out_path)

        output_text(test_model, valid_input, vocab_char)

if __name__ == '__main__':
    main(sys.argv[1:])
