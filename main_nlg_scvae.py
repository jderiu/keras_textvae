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
from data_loaders.data_loader_nlg import load_text_gen_data
from vae_gan_architectures.sc_vae import get_vae_gan_model
import time
from custom_callbacks import StepCallback, GANOutputCallback, TerminateOnNaN
from keras_fit_utils.utils import _make_batches, _batch_shuffle, _slice_arrays


def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def clip_weights(model):
    for layer in model.layers:
        weights = layer.get_weights()
        weights = [np.clip(w, -0.5, 0.5) for w in weights]
        layer.set_weights(weights)


def sample_noise(noise_scale, batch_size, noise_dim):
    return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim[0]))


def gen_batch(X, batch_size):
    while True:
        idx = np.random.choice(X.shape[0], batch_size, replace=False)
        yield idx


def get_disc_batch(X_real_batch, generator_model, batch_counter, batch_size, noise_dim, noise_scale=0.5):
    # Pass noise to the generator
    noise_input = sample_noise(noise_scale, batch_size, noise_dim)
    # Produce an output
    X_disc_gen = generator_model.predict(noise_input, batch_size=batch_size)
    X_disc_real = X_real_batch[:batch_size]

    return X_disc_real, X_disc_gen

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

        #== == == == == == =
        # Load all the Data
        #== == == == == == =

        noutputs = 12

        logging.info('Load Training Data')
        train_input, train_output, train_lex = load_text_gen_data(join(tweets_path, 'trainset.csv'), config_data, vocab,noutputs, word_based=False)
        logging.info('Load Validation Data')
        valid_input, valid_output, valid_lex = load_text_gen_data(join(tweets_path, 'devset.csv'), config_data, vocab,noutputs, word_based=False)
        logging.info('Load Output Validation Data')
        valid_dev_input, valid_dev_output, valid_dev_lex = load_text_gen_data(join(tweets_path, 'devset_reduced.csv'), config_data, vocab, noutputs, random_output=False, word_based=False)

        #train_input = [x[:1213] for x in train_input]
        #train_output = [x[:1213] for x in train_output]

        noise_valid_input = np.zeros(shape=(valid_input[0].shape[0], config_data['z_size']))

        step = K.variable(1.)
        steps_per_epoch = ceil(train_output[0].shape[0] / config_data['batch_size'])
        # == == == == == == == == == == =
        # Define and load the CNN model
        # == == == == == == == == == == =
        vae_model_train, vae_model_test, vae_vanilla_train_model, vae_vanilla_test_model, discriminator_model, decoder_test, discriminator = get_vae_gan_model(config_data, vocab, step)
        with open(os.path.join(log_path, 'models.txt'), 'wt') as fh:
            fh.write('VAE Model Train\n')
            fh.write('---------\n')
            vae_model_train.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write('VAE Model Test\n')
            fh.write('--------------\n')
            vae_model_test.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write('VAE Model Pretrain\n')
            fh.write('---------------------------\n')
            vae_vanilla_train_model.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write('VAE Model Pretrain Test\n')
            fh.write('---------------------------\n')
            vae_vanilla_test_model.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write('Decoder Test\n')
            fh.write('-------------------\n')
            decoder_test.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write('Discriminator Models\n')
            fh.write('-------------------\n')
            discriminator_model.summary(print_fn=lambda x: fh.write(x + '\n'))

        terminate_on_nan = TerminateOnNaN()
        model_checkpoint = ModelCheckpoint('models/vae_model/weights.{epoch:02d}.hdf5', period=10, save_weights_only=True)

        out_labels = ['enc_' + s for s in vae_model_train._get_deduped_metrics_names()]
        out_labels += ['dis_' + s for s in discriminator_model._get_deduped_metrics_names()]

        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        step_callback = StepCallback(step, steps_per_epoch)
        output_callback = GANOutputCallback(vae_model_train, valid_dev_input, 1, vocab, '', fname='{}/test_output'.format(log_path))
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

        steps = 0
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
                enc_outs = vae_model_train.train_on_batch(x=X, y=y[:12])

                set_trainability(discriminator, trainable=True)
                list_disc_loss_real = []
                if steps < 25 or steps % 500 == 0:
                    disc_iterations = 5
                else:
                    disc_iterations = discriminator_iterations
                for disc_it in range(disc_iterations):
                    real_idx = np.random.choice(train_input[0].shape[0], len(batch_ids), replace=False)

                    disX_train = train_input[-1][real_idx] #take input 8 as train input and the rest as targets
                    disy_train = [x[real_idx] for x in train_input[:8]] #take input 1-7 as targets

                    #train on real data
                    dis_outs_real = discriminator_model.train_on_batch(disX_train, disy_train)

                    list_disc_loss_real.append(dis_outs_real)

                loss_d_real = np.mean(list_disc_loss_real, axis=0)

                outs = np.concatenate((enc_outs, loss_d_real))

                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)
                epoch_logs = {}
                batch_index += 1
                steps_done += 1
                steps += 1
                # Epoch finished.
                if steps_done >= steps_per_epoch:
                    valid_len = valid_output[0].shape[0]
                    enc_val_outs = vae_model_train.evaluate(valid_input, valid_output[:12], verbose=False)
                    dis_val_outs = discriminator_model.evaluate(valid_input[-1], valid_input[:8], verbose=False)

                    val_outs = enc_val_outs + dis_val_outs

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
