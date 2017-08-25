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


from keras.callbacks import BaseLogger, ProgbarLogger, CallbackList
from custom_callbacks import MultiModelCheckpoint, TerminateOnNaN, OutputCallback, StepCallback
from output_text import output_text
from data_loaders.data_loader_hybrid import load_data, generate_data_stream
from vae_gan_architectures.hybrid_gan_model import get_vae_gan_model
from keras.utils.data_utils import GeneratorEnqueuer


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

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename='logging/vae_gan/evolution.log', filemode=filemode)

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

        noutputs = 5
        logging.info('Load Validation Data')
        valid_input, valid_output = load_data(join(tweets_path, 'en_valid15.tsv'), config_data, vocab_word, vocab_char, noutputs)

        step = K.variable(1.)

        # == == == == == == == == == == =
        # Define and load the CNN model
        # == == == == == == == == == == =
        vae_train_model, vae_gan_model, discriminator_model, discriminators, vae_test_model = get_vae_gan_model(config_data, vocab_char, step)
        #full_model.summary()
        vae_train_model.summary()
        vae_gan_model.summary()
        discriminator_model.summary()

        model_path = 'models/vae_model/'
        steps_per_epoch = int(ceil(config_data['samples_per_epoch'] / config_data['batch_size']))
        epochs = int(ceil(config_data['nb_epochs']*(config_data['nsamples']/config_data['samples_per_epoch'])))
        batch_size = config_data['batch_size']

        initial_epoch = 0
        skip_texts = 0

        save_models = [
            (vae_train_model, 'models/vae_model/vae_train_weights.hdf5'),
            (vae_gan_model, 'models/vae_model/vae_gan_weights.hdf5'),
            (discriminator_model, 'models/vae_model/discriminator_weights.hdf5'),
            (vae_test_model, 'models/vae_modelvae_test_weights.hdf5'),
        ]

        terminate_on_nan = TerminateOnNaN()
        model_checkpoint = MultiModelCheckpoint(models=save_models, period=15)

        generator = generate_data_stream(config_data['training_path'], config_data, vocab_char, vocab_word, config_data['batch_size'], skip_data=skip_texts, noutputs=noutputs)

        enc_out_labels = ['vae_train_' + s for s in vae_train_model._get_deduped_metrics_names()]
        dec_out_labels = ['vae_gan_' + s for s in vae_gan_model._get_deduped_metrics_names()]
        dis_out_labels = ['dis_' + s for s in discriminator_model._get_deduped_metrics_names()]
        out_labels = enc_out_labels + dec_out_labels + dis_out_labels

        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        step_callback = StepCallback(step, steps_per_epoch)
        output_callback = OutputCallback(vae_test_model, valid_input, 15, vocab_char, '')
        callbacks = CallbackList([BaseLogger(), ProgbarLogger(count_mode='steps'), step_callback, output_callback, model_checkpoint])

        callbacks.set_model(vae_gan_model)
        callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': True,
            'do_validation': True,
            'metrics': callback_metrics,
        })

        enqueuer = GeneratorEnqueuer(generator, use_multiprocessing=False, wait_time=0.01)
        try:
            enqueuer.start(workers=1, max_queue_size=10)
            output_generator = enqueuer.get()
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

                    set_trainability(vae_train_model, trainable=True)
                    for discriminator in discriminators:
                        set_trainability(discriminator, trainable=False)
                    vae_train_outs = vae_train_model.train_on_batch(X, y)
                    vae_gan_outs = vae_gan_model.train_on_batch(X, y[:3])

                    set_trainability(vae_train_model, trainable=False)
                    for discriminator in discriminators:
                        set_trainability(discriminator, trainable=True)

                    dis_outs = discriminator_model.train_on_batch(X, y[:3])
                    outs = vae_train_outs + vae_gan_outs + dis_outs

                    # outs = full_model.train_on_batch(X, y)

                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o

                    callbacks.on_batch_end(batch_index, batch_logs)
                    epoch_logs = {}
                    batch_index += 1
                    steps_done += 1

                    # Epoch finished.
                    if steps_done >= steps_per_epoch:
                        vae_train_val_outs = vae_train_model.evaluate(valid_input, valid_output, verbose=False)
                        vae_gan_val_outs = vae_gan_model.evaluate(valid_input, valid_output[:3], verbose=False)
                        dis_val_outs = discriminator_model.evaluate(valid_input, valid_output[:3], verbose=False)

                        val_outs = vae_train_val_outs + vae_gan_val_outs + dis_val_outs

                        # val_outs = full_model.evaluate(valid_input, valid_output, verbose=False)

                        if not isinstance(val_outs, list):
                            val_outs = [val_outs]
                        # Same labels assumed.
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o

                callbacks.on_epoch_end(epoch, epoch_logs)
                epoch += 1
        finally:
            if enqueuer is not None:
                enqueuer.stop()


        callbacks.on_train_end()

if __name__ == '__main__':
    main(sys.argv[1:])
