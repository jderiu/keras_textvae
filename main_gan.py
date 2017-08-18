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


from keras.callbacks import Callback, ModelCheckpoint, BaseLogger, ProgbarLogger, CallbackList
from keras.optimizers import Adam, Nadam, Adadelta
from output_text import output_text
from data_loaders.data_loader_charlevel import load_data, generate_data_stream
from vae_gan_architectures.vae_gan_deconv_recurrent import vae_gan_model
from tqdm import tqdm
from keras.utils.data_utils import GeneratorEnqueuer


class NewCallback(Callback):
    def __init__(self, alpha, steps_per_epoch):
        self.alpha = alpha
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0
        super(NewCallback, self).__init__()

    def on_batch_end(self, batch, logs=None):
        value = self.steps_per_epoch*self.current_epoch + batch
        K.set_value(self.alpha, value)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch


class OutputCallback(Callback):
    def __init__(self, test_model, validation_input, frequency, vocabulary, delimiter):
        self.validation_input = validation_input
        self.vocabulary = vocabulary
        self.test_model = test_model
        self.frequency = frequency
        self.delimiter = delimiter
        super(OutputCallback, self).__init__()

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.frequency == 0:
            output_text(self.test_model, self.validation_input, self.vocabulary, str(epoch), delimiter=self.delimiter)

    def on_epoch_end(self, epoch, logs={}):
        self.ep_end_weights = {}

        enc_loss = logs.get('enc_loss', '-')
        dec_loss = logs.get('dec_loss', '-')
        dis_loss = logs.get('dis_loss', '-')
        val_enc_loss = logs.get('val_enc_loss', '-')
        val_dec_loss = logs.get('val_dec_loss', '-')
        val_dis_loss = logs.get('val_dis_loss', '-')

        logging.info('TRAINING: Enc Loss: {}\t Dec Loss: {}\tDis Loss: {}'.format(enc_loss, dec_loss, dis_loss))
        logging.info('VALIDATION: Enc Loss: {}\t Dec Loss: {}\tDis Loss: {}'.format(val_enc_loss, val_dec_loss, val_dis_loss))
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


def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def pretrain_discriminator(model, data, vocab):
    nsamples = data.shape[0]
    sample_size = data.shape[1]
    y_orig = np.ones((nsamples, ))
    y_fake = np.zeros((nsamples, ))

    fake_data = np.random.randint(low=0, high=max(vocab.values()), size=(nsamples, sample_size))
    sen_lens = np.random.normal(loc=60, scale=20, size=nsamples)


    train_set = np.vstack((data, fake_data))
    labels = np.vstack((y_orig, y_fake))

    model.fit(train_set, labels, epochs=20)


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
                        filename='logging/vae_gan/evolution.log', filemode=filemode)

    with open(config_fname, 'r') as json_data:
        config_data = json.load(json_data)

        tweets_path = config_data['tweets_path']
        vocab_path = config_data['vocab_path']
        vocab = cPickle.load(open(join(vocab_path, 'vocabulary.pkl'), 'rb'))

        #== == == == == == =
        # Load all the Data
        #== == == == == == =

        noutputs = 5

        logging.info('Load Training Data')
        train_input, train_output = load_data(join(tweets_path, 'en_train.tsv'),   config_data, vocab, noutputs)
        logging.info('Load Validation Data')
        valid_input, valid_output = load_data(join(tweets_path, 'en_valid15.tsv'), config_data, vocab, noutputs)
        logging.info('Load Validation Data')
        valid_input2, valid_output2 = load_data(join(tweets_path, 'en_test16.tsv'),  config_data, vocab, noutputs)
        logging.info('Load Test Data')
        test_input, test_output = load_data(join(tweets_path, 'en_test17.tsv'),  config_data, vocab, noutputs)

        step = K.variable(1.)

        # == == == == == == == == == == =
        # Define and load the CNN model
        # == == == == == == == == == == =
        full_model, encoding_train_model, decoder_train_model, discriminator_train_model, decoder_inference, encoder, decoder, discriminator, discriminator_pretrain_model = vae_gan_model(config_data, vocab, step)
        #full_model.summary()
        encoding_train_model.summary()
        decoder_train_model.summary()
        discriminator_train_model.summary()
        decoder_inference.summary()
        encoder.summary()
        decoder.summary()
        discriminator.summary()

        #pretrain_discriminator(discriminator_pretrain_model, train_input, vocab)

        model_path = 'models/vae_model/'
        steps_per_epoch = int(ceil(config_data['samples_per_epoch'] / config_data['batch_size']))
        epochs = int(ceil(config_data['nb_epochs']*(config_data['nsamples']/config_data['samples_per_epoch'])))
        batch_size = config_data['batch_size']

        initial_epoch = 0
        skip_texts = 0

        terminate_on_nan = TerminateOnNaN()
        model_checkpoint = ModelCheckpoint('models/vae_model/weights.{epoch:02d}.hdf5', period=10, save_weights_only=True)

        generator = generate_data_stream(config_data['training_path'], config_data, vocab, config_data['batch_size'], skip_data=skip_texts, noutputs=noutputs)
        enqueuer = GeneratorEnqueuer(generator, use_multiprocessing=False, wait_time=0.01)
        enqueuer.start(workers=1, max_queue_size=10)
        output_generator = enqueuer.get()

        enc_out_labels = ['enc_' + s for s in encoding_train_model._get_deduped_metrics_names()]
        dec_out_labels = ['dec_' + s for s in decoder_train_model._get_deduped_metrics_names()]
        dis_out_labels = ['dis_' + s for s in discriminator_train_model._get_deduped_metrics_names()]
        out_labels = enc_out_labels + dec_out_labels + dis_out_labels

        #out_labels = full_model._get_deduped_metrics_names()

        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        step_callback = NewCallback(step, steps_per_epoch)
        output_callback = OutputCallback(decoder_inference, valid_input, 15, vocab, '')
        callbacks = CallbackList([BaseLogger(), ProgbarLogger(count_mode='steps'), step_callback, output_callback])

        callbacks.set_model(full_model)
        callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': True,
            'do_validation': True,
            'metrics': callback_metrics,
        })

        callbacks.on_train_begin()

        epoch = initial_epoch
        while epoch < epochs:
            epoch_logs = {}
            callbacks.on_epoch_begin(epoch)
            steps_done = 0
            batch_index = 0
            while steps_done < steps_per_epoch:
                batch_logs = {}

                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size

                X, y = next(output_generator)

                callbacks.on_batch_begin(batch_index, batch_logs)

                set_trainability(encoder, trainable=True)
                set_trainability(decoder, trainable=False)
                set_trainability(discriminator, trainable=False)
                enc_outs = encoding_train_model.train_on_batch(X, y[:3])

                set_trainability(encoder, trainable=False)
                set_trainability(decoder, trainable=True)
                set_trainability(discriminator, trainable=False)
                dec_outs = decoder_train_model.train_on_batch(X, y[:4])

                set_trainability(encoder, trainable=False)
                set_trainability(decoder, trainable=False)
                set_trainability(discriminator, trainable=True)

                dis_outs = discriminator_train_model.train_on_batch(X, y[0])
                outs = enc_outs + dec_outs + [dis_outs]

                #outs = full_model.train_on_batch(X, y)

                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)
                epoch_logs = {}
                batch_index += 1
                steps_done += 1

                # Epoch finished.
                if steps_done >= steps_per_epoch:
                    enc_val_outs = encoding_train_model.evaluate(valid_input, valid_output[:3], verbose=False)
                    dec_val_outs = decoder_train_model.evaluate(valid_input, valid_output[:4], verbose=False)
                    dis_val_outs = discriminator_train_model.evaluate(valid_input, valid_output[0], verbose=False)

                    val_outs = enc_val_outs + dec_val_outs + [dis_val_outs]

                    #val_outs = full_model.evaluate(valid_input, valid_output, verbose=False)

                    if not isinstance(val_outs, list):
                        val_outs = [val_outs]
                    # Same labels assumed.
                    for l, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + l] = o

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1

        callbacks.on_train_end()

if __name__ == '__main__':
    main(sys.argv[1:])
