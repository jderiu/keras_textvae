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
from vae_architectures.vae_deconv_recurrent_nlg import vae_model
from data_loaders.data_loader_nlg import load_text_gen_data

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
        self.validation_input = validation_input[0]
        self.vocabulary = vocabulary
        self.test_model = test_model
        self.frequency = frequency
        self.delimiter = delimiter
        super(OutputCallback, self).__init__(**kwargs)

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.frequency == 0:
            output_text(self.test_model, self.validation_input, self.vocabulary, str(epoch), delimiter=self.delimiter)

        model = self.model
        self.ep_begin_weights = {}
        for layer in model.layers:
            if layer.trainable:
                name = layer.name
                self.ep_begin_weights[name] = layer.get_weights()

    def on_epoch_end(self, epoch, logs={}):
        model = self.model
        self.ep_end_weights = {}

        for layer in model.layers:
            if layer.trainable:
                name = layer.name
                self.ep_end_weights[name] = layer.get_weights()

        logging.info('Layer Deltas of Epoch: {}'.format(epoch))
        #compute norms etc
        for layer_name in self.ep_begin_weights.keys():
            bweights = self.ep_begin_weights[layer_name]
            eweights = self.ep_end_weights[layer_name]

            for bweight, eweight in zip(bweights, eweights):
                delta = eweight - bweight
                layer_delta = np.linalg.norm(delta)
                logging.info('\tLayer Deltas of Layer {}: {}'.format(layer_name, layer_delta))

        main_loss = logs.get('main_loss_loss', '-')
        kld_loss = logs.get('kld_loss_loss', '-')
        auxiliary_loss = logs.get('auxiliary_loss_loss', '-')
        val_main_loss = logs.get('val_main_loss_loss', '-')
        val_kld_loss = logs.get('val_kld_loss_loss', '-')
        val_auxiliary_loss = logs.get('val_auxiliary_loss_loss', '-')

        logging.info('TRAINING: Total Loss: {}\t Main Loss: {}\tKLD Loss: {}\tAuxiliary Loss: {}'.format(logs['loss'], main_loss, kld_loss, auxiliary_loss))
        logging.info('VALIDATION: Total Loss: {}\t Main Loss: {}\tKLD Loss: {}\tAuxiliary Loss: {}'.format(logs['val_loss'], val_main_loss, val_kld_loss, val_auxiliary_loss))
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
        vocab_path = config_data['vocab_path']
        vocab = cPickle.load(open(join(vocab_path, 'vocabulary.pkl'), 'rb'))

        #== == == == == == =
        # Load all the Data
        #== == == == == == =

        delimiter = ''
        noutputs = 3

        logging.info('Load Training Data')
        train_input, train_output = load_text_gen_data(join(tweets_path, 'trainset.csv'),   config_data, vocab, noutputs)
        logging.info('Load Validation Data')
        valid_input, valid_output = load_text_gen_data(join(tweets_path, 'devset.csv'), config_data, vocab, noutputs)

        step = K.variable(1.)

        # == == == == == == == == == == =
        # Define and load the CNN model
        # == == == == == == == == == == =
        cnn_model, test_model = vae_model(config_data, vocab, step)
        cnn_model.save_weights(config_data['base_model_path'])
        cnn_model.summary()

        model_path = 'models/vae_model/'
        steps_per_epoch = ceil(config_data['samples_per_epoch'] / config_data['batch_size'])

        terminate_on_nan = TerminateOnNaN()
        model_checkpoint = ModelCheckpoint('models/vae_model/weights.{epoch:02d}.hdf5', period=10, save_weights_only=True)
        reduce_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.995, patience=100, min_lr=0.001, cooldown=50)

        optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.001, clipnorm=10)
        #optimizer = Adadelta(lr=1.0, epsilon=1e-8, rho=0.95, decay=0.0001, clipnorm=10)
        cnn_model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)

        cnn_model.fit(
            x=train_input,
            y=train_output,
            epochs=1000,
            batch_size=config_data['batch_size'],
            validation_data=(valid_input, valid_output),
            callbacks=[NewCallback(step, steps_per_epoch),
                       OutputCallback(test_model, valid_input, 15, vocab, delimiter), terminate_on_nan,
                       model_checkpoint, reduce_callback],
        )

        test_model.summary()

        cnn_out_path = join(config_data['output_path'], 'trained_deconv_vae_{}_model'.format(config_data['model_type']))
        cnn_model.save_weights(cnn_out_path)
        cnn_out_path = join(config_data['output_path'], 'trained_deconv_vae_{}_model_test'.format(config_data['model_type']))
        test_model.save_weights(cnn_out_path)

        output_text(test_model, valid_input, vocab)

if __name__ == '__main__':
    main(sys.argv[1:])
