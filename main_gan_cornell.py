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


from keras.callbacks import ModelCheckpoint, BaseLogger, ProgbarLogger, CallbackList
from data_loaders.data_loader_charlevel import load_text_pairs
from vae_gan_architectures.vae_gan_cornell import get_vae_gan_model
import time
from custom_callbacks import StepCallback, OutputCallback, TerminateOnNaN
from keras_fit_utils.utils import _make_batches, _batch_shuffle, _slice_arrays

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


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
        tweets_path = config_data['tweets_path']
        vocab_path = config_data['vocab_path']
        vocab = cPickle.load(open(join(vocab_path, 'vocabulary.pkl'), 'rb'))

        #== == == == == == =
        # Load all the Data
        #== == == == == == =

        noutputs = 5

        logging.info('Load Training Data')
        train_input, train_output = load_text_pairs(join(tweets_path, 'training_set.tsv'), config_data, vocab, noutputs)
        logging.info('Load Validation Data')
        valid_input, valid_output = load_text_pairs(join(tweets_path, 'vaild_set.tsv'), config_data, vocab, noutputs)
        logging.info('Load Output Validation Data')
        valid_dev_input, valid_dev_output = load_text_pairs(join(tweets_path, 'test_set.tsv'), config_data, vocab,noutputs)

        noise_valid_input = np.zeros(shape=(valid_input[0].shape[0], config_data['z_size']))

        step = K.variable(1.)
        steps_per_epoch = ceil(train_output[0].shape[0] / config_data['batch_size'])
        # == == == == == == == == == == =
        # Define and load the CNN model
        # == == == == == == == == == == =
        vae_model, vae_model_test, decoder_discr_model, discriminator_model, discriminator = get_vae_gan_model(config_data, vocab, step)
        with open(os.path.join(log_path, 'models.txt'), 'wt') as fh:
            fh.write('VAE Model\n')
            fh.write('---------\n')
            vae_model.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write('VAE Model Test\n')
            fh.write('--------------\n')
            vae_model_test.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write('Decoder Discriminator Model\n')
            fh.write('---------------------------\n')
            decoder_discr_model.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write('Discriminator Model\n')
            fh.write('-------------------\n')
            discriminator_model.summary(print_fn=lambda x: fh.write(x + '\n'))

        terminate_on_nan = TerminateOnNaN()
        model_checkpoint = ModelCheckpoint('models/vae_model/weights.{epoch:02d}.hdf5', period=10, save_weights_only=True)

        enc_out_labels = ['enc_' + s for s in vae_model._get_deduped_metrics_names()]
        dec_out_labels = ['dec_' + s for s in decoder_discr_model._get_deduped_metrics_names()]
        dis_out_labels = ['dis_' + s for s in discriminator_model._get_deduped_metrics_names()]
        out_labels = enc_out_labels + dec_out_labels + dis_out_labels

        #out_labels = full_model._get_deduped_metrics_names()

        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        step_callback = StepCallback(step, steps_per_epoch)
        output_callback = OutputCallback(vae_model_test, valid_dev_input[0], 1, vocab, '', fname='{}/test_output'.format(log_path))
        callbacks = CallbackList([BaseLogger(), ProgbarLogger(count_mode='steps'), step_callback, output_callback, model_checkpoint, terminate_on_nan])

        callbacks.set_model(vae_model_test)
        callbacks.set_params({
            'batch_size': batch_size,
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': True,
            'do_validation': True,
            'metrics': callback_metrics or [],
        })

        callbacks.on_train_begin()
        initial_epoch = 0
        num_train_samples = train_input[0].shape[0]
        index_array = np.arange(num_train_samples)

        epoch = initial_epoch
        while epoch < epochs:
            epoch_logs = {}
            callbacks.on_epoch_begin(epoch)
            index_array = _batch_shuffle(index_array, batch_size)

            steps_done = 0
            batches = _make_batches(num_train_samples, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_logs = {}
                batch_ids = index_array[batch_start:batch_end]
                X = _slice_arrays(train_input, batch_ids)
                y = _slice_arrays(train_output, batch_ids)

                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size

                callbacks.on_batch_begin(batch_index, batch_logs)

                set_trainability(discriminator, trainable=False)
                enc_outs = vae_model.train_on_batch(x=X, y=y[:3])

                noise_input = np.zeros(shape=(batch_size, config_data['z_size']))
                decoder_discr_input = [X[0], noise_input]
                dec_outs = decoder_discr_model.train_on_batch(x=decoder_discr_input, y=-np.ones(shape=(batch_size, 1)))

                set_trainability(discriminator, trainable=True)
                #train on real data
                x_fake = vae_model_test.predict_on_batch(x=X[0])


                x_discr_training_inputs = X[0] + X[0]
                x_discr_training_outputs = X[1] + x_fake
                x_discr = [x_discr_training_inputs, x_discr_training_outputs]
                y_discr_training = np.concatenate(
                    (
                        -np.ones(shape=(batch_size, 1)),
                        np.ones(shape=(batch_size, 1))
                    ))

                dis_outs = discriminator_model.train_on_batch(x_discr, y_discr_training)
                outs = enc_outs + dec_outs + [dis_outs]

                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)
                epoch_logs = {}
                batch_index += 1
                steps_done += 1

                # Epoch finished.
                if steps_done >= steps_per_epoch:
                    valid_len = valid_output[0].shape[0]
                    enc_val_outs = vae_model.evaluate(valid_input, valid_output[:3], verbose=False)
                    dec_val_outs = decoder_discr_model.evaluate([valid_input[0], noise_valid_input], -np.ones(shape=(valid_len, 1)), verbose=False)
                    dis_val_outs = discriminator_model.evaluate(valid_input, -np.ones(shape=(valid_len, 1)), verbose=False)

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
