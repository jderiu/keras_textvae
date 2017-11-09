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


from keras.callbacks import ModelCheckpoint, BaseLogger, ProgbarLogger, CallbackList, TensorBoard,EarlyStopping
from data_loaders.data_loader_nlg import load_text_gen_data
from sc_lstm_architecutre.sclstm_gan_architecture import vae_model
import time
from keras.optimizers import Adam, Nadam, Adadelta
from custom_callbacks import StepCallback, GANOutputCallback, TerminateOnNaN, LexOutputCallbackGAN
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

        model_path = config_data['output_path']
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        #== == == == == == =
        # Load all the Data
        #== == == == == == =
        delimiter = ''
        noutputs = 10

        logging.info('Load Training Data')
        train_input, train_output, train_weights, train_lex = load_text_gen_data(join(tweets_path, 'trainset.csv'), config_data, vocab,noutputs, word_based=False)
        logging.info('Load Validation Data')
        valid_input, valid_output, _, valid_lex = load_text_gen_data(join(tweets_path, 'devset.csv'), config_data, vocab,noutputs, word_based=False)
        logging.info('Load Output Validation Data')
        valid_dev_input, valid_dev_output, _, valid_dev_lex = load_text_gen_data(join(tweets_path, 'devset_reduced.csv'), config_data, vocab, noutputs, random_output=True, word_based=False)
        valid_dev_input2, valid_dev_output2, _, valid_dev_lex2 = load_text_gen_data(join(tweets_path, 'devset_located.csv'), config_data, vocab, noutputs, random_output=True, word_based=False,random_first_word=True)
        valid_dev_input3, valid_dev_output3, _, valid_dev_lex3 = load_text_gen_data(join(tweets_path, 'test_e2e.csv'), config_data, vocab, noutputs, random_output=True, word_based=False,random_first_word=True)

        step = K.variable(1., name='step_varialbe')
        steps_per_epoch = ceil(train_output[0].shape[0] / config_data['batch_size'])
        # == == == == == == == == == == =
        # Define and load the CNN model
        # == == == == == == == == == == =
        train_model, test_model, discriminator_models = vae_model(config_data, vocab, step)
        with open(os.path.join(log_path, 'models.txt'), 'wt') as fh:
            fh.write('VAE Model Train\n')
            fh.write('---------\n')
            train_model.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write('VAE Model Test\n')
            fh.write('--------------\n')
            test_model.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write('Discriminator Models\n')
            for discriminator in discriminator_models:
                fh.write('{}\n'.format(discriminator.name))
                fh.write('---------------------------\n')
                discriminator.summary(print_fn=lambda x: fh.write(x + '\n'))

        optimizer = Adadelta(lr=1, epsilon=1e-8, rho=0.95, decay=0.0001, clipnorm=10)
        train_model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred, )
        disX_train = train_input[-1]  # take input 8 as train input and the rest as targets
        disy_train = train_input[:9]  # take input 1-7 as targets
        if config_data.get('pretrain_dirscr', 1) == 1:
            for i, discriminator in enumerate(discriminator_models):
                discriminator.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']) #output of the discriminator model are the outputs -> specifiy cross_entropy as loss

                # == == == == == == == == =
                # Pretrain Discriminators
                # == == == == == == == == =
                early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, min_delta=1e-6)
                model_checkpoint = ModelCheckpoint(join(model_path, 'discr_weights_{}.hdf5'.format(discriminator.name)), save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
                logging.info('Pretrain the {} Discriminator'.format(discriminator.name))
                history = discriminator.fit(
                    x=disX_train,
                    y=disy_train[i],
                    validation_data=(valid_input[-1], valid_input[i]),
                    epochs=1000,
                    batch_size=batch_size,
                    callbacks=[early_stopping, model_checkpoint]
                )

                losses = history.history['loss']
                val_losses = history.history['val_loss']
                val_accs = history.history['val_acc']

                for i, (loss, val_loss, val_acc) in enumerate(zip(losses, val_losses, val_accs)):
                    logging.info('Epoch: {} Loss: {} Val Loss: {} Val Acc: {}'.format(i, loss, val_loss, val_acc))

        for i, discriminator in enumerate(discriminator_models):
            logging.info('Loading the {} Discriminator'.format(discriminator.name))
            model_weights = join(model_path, 'discr_weights_{}.hdf5'.format(discriminator.name))
            discriminator.load_weights(model_weights)

        terminate_on_nan = TerminateOnNaN()
        model_checkpoint = ModelCheckpoint(join(model_path, 'weights.{epoch:02d}.hdf5'), period=15, save_weights_only=True)

        tensorboard = TensorBoard(log_dir='logging/tensorboard', histogram_freq=0, write_grads=True, write_images=True)
        step_callback = StepCallback(step, steps_per_epoch)
        lex_output = LexOutputCallbackGAN(test_model, valid_dev_input, valid_dev_lex, 1, vocab, delimiter, fname='{}/test_output'.format(log_path))
        lex_output2 = LexOutputCallbackGAN(test_model, valid_dev_input2, valid_dev_lex2, 1, vocab, delimiter, fname='{}/test_output_random'.format(log_path))
        lex_output3 = LexOutputCallbackGAN(test_model, valid_dev_input3, valid_dev_lex3, 1, vocab, delimiter, fname='{}/final_test_output_random'.format(log_path))
        callbacks = [step_callback, tensorboard, lex_output, lex_output2, lex_output3, model_checkpoint, terminate_on_nan]
        for i, discriminator in enumerate(discriminator_models):
            set_trainability(discriminator, trainable=False)

        logging.info('Train the Model')

        train_model.fit(
            x=train_input,
            y=train_output,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(valid_input, valid_output),
            sample_weight=train_weights,
            callbacks=callbacks
        )

if __name__ == '__main__':
    main(sys.argv[1:])
