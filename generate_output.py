# -*- coding: utf-8 -*-

import _pickle as cPickle
import json
import logging
from collections import defaultdict
from os.path import join
import random
import numpy as np
#do this before importing anything from Keras
np.random.seed(1337)
import keras.backend as K
from data_loaders.data_loader_nlg import load_text_gen_data, get_fist_words_for_input
from sc_lstm_architecutre.sclstm_gan_architecture import vae_model, get_discriminator_models
from sklearn.metrics import accuracy_score


consonants = ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Z']

def produce_output(test_model, discriminator_models, inputs, input_lex, inverse_vocab, overlap_map_for_fw):
    def upperfirst(x):
        return x[0].upper() + x[1:]

    name_tok = 'XNAMEX'
    near_tok = 'XNEARX'
    food_tok = 'XFOODX'

    test_inputs = []
    test_input_indices = []
    fw_scores = []
    for idx, input in enumerate(zip(*inputs + [input_lex[name_tok], input_lex[near_tok], input_lex[food_tok]])):
        first_word = input[8]
        fw_incides = get_fist_words_for_input(input[:8], overlap_map_for_fw)
        for i, fw_score in fw_incides:
            new_first_word = np.zeros_like(first_word)
            new_first_word[i] = 1.0

            ninput = list(input[:10])
            ninput[8] = new_first_word
            test_inputs.append(ninput)
            test_input_indices.append(idx)
            fw_scores.append(fw_score)

    correct_test_inputs_dict = defaultdict(lambda: [])
    for test_input in test_inputs:
        for i, iput in enumerate(test_input):
            correct_test_inputs_dict[i].append(iput)

    correct_test_inputs = [[]]*10
    for i, values in sorted(correct_test_inputs_dict.items(), key=lambda x: x[0]):
        correct_test_inputs[i] = np.asarray(correct_test_inputs_dict[i])

    logging.info('Predicting sentences using SCLSTM')
    sentences = test_model.predict(correct_test_inputs, batch_size=1024, verbose=1)
    scores = []
    for i, discriminator in enumerate(discriminator_models):
        discriminator.compile(optimizer='adadelta', loss='categorical_crossentropy')
        logging.info('Computing Score of {} Discriminator'.format(discriminator.name))
        y = np.argmax(correct_test_inputs[i], axis=1)
        r = np.arange(start=0, stop=y.shape[0])
        y_pred = discriminator.predict(x=sentences, batch_size=1024, verbose=1)
        y_pred_argmax = np.argmax(y_pred, axis=1)
        correct_lbl = y == y_pred_argmax
        print(accuracy_score(y, y_pred_argmax))
        print(correct_lbl.shape)
        scores.append(correct_lbl.astype(int)*y_pred[r, y])
        #scores.append(y_pred[r, y])

    sen_dict = defaultdict(lambda: [])
    print(len(test_input_indices))
    print(len(sentences))
    print(len([sum(x) for x in zip(*scores)]))
    ofile = open('output.txt', 'wt', encoding='utf-8')
    gen_ofile = open('generated_output_devset_689.txt', 'wt', encoding='utf-8')
    for test_input_idx, sentence, score in zip(test_input_indices, sentences, [sum(x) for x in zip(*scores)]):

        list_txt_idx = [int(x) for x in sentence.tolist()]
        txt_list = [inverse_vocab.get(int(x), '') for x in list_txt_idx]
        oline = ''.join(txt_list)
        for lex_key in input_lex.keys():
            val = input_lex[lex_key][test_input_idx]
            if val:
                oline = oline.replace(lex_key, val)

        if near_tok in oline or name_tok in oline or food_tok in oline:
            continue
        sen_dict[test_input_idx].append((oline, score))

    for i, sentences in sen_dict.items():
        sorted_sentences = sorted(sentences, key=lambda x: x[1], reverse=True)
        for sentence, score in sorted(sentences, key=lambda x: x[1], reverse=True):
            ofile.write(upperfirst('{}\t{}\n'.format(sentence, score)))
        ofile.write('\n')

        max_score = max([x[1] for x in sentences])
        max_score_sentences = [x[0] for x in sentences if x[1] > 8.0]
        if len(max_score_sentences) > 0:
            sample_sentence = random.choice(max_score_sentences)
        else:
            sample_sentence = random.choice(sorted_sentences[:5])[0]
        for consonant in consonants:
            sample_sentence = sample_sentence.replace('An {}'.format(consonant), 'A {}'.format(consonant))

        sample_sentence = sample_sentence.replace('Fast food food', 'Fast food')
        sample_sentence = sample_sentence.replace('The The', 'The')
        gen_ofile.write(upperfirst(sample_sentence) + '\n')

    return sentences

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_path = 'models/sclstm_gan_128filter'

config_data = json.load(open('configurations/config_vae_scvae.json', 'r'))

tweets_path = 'en_full'
vocab_path = 'en_full'
vocab = cPickle.load(open(join(vocab_path, 'vocabulary.pkl'), 'rb'))
overlap_map_for_fw = cPickle.load(open(join(vocab_path, 'overlap_map_for_fw.pkl'), 'rb'))
inverse_vocab = {v: k for (k, v) in vocab.items()}

valid_dev_input2, valid_dev_output2, _, valid_dev_lex2 = load_text_gen_data(join(tweets_path, 'devset_located.csv'),
                                                                            config_data, vocab, 10,
                                                                            random_output=True, word_based=False,
                                                                            random_first_word=True)
valid_dev_input3, valid_dev_output3, _, valid_dev_lex3 = load_text_gen_data(join(tweets_path, 'test_e2e.csv'),
                                                                            config_data, vocab, 10,
                                                                            random_output=True, word_based=False,
                                                                            random_first_word=True)

step = K.variable(1., name='step_varialbe')
train_model, test_model, _ = vae_model(config_data, vocab, step)
discriminator_models = get_discriminator_models(config_data, vocab)


logging.info('Loading the SCLSTM Model')
train_model.load_weights(join(model_path, 'weights.689.hdf5'))

for i, discriminator in enumerate(discriminator_models):
    logging.info('Loading the {} Discriminator'.format(discriminator.name))
    discriminator.load_weights(join(model_path, 'discr_weights_{}.hdf5'.format(discriminator.name)))


produce_output(test_model, discriminator_models, valid_dev_input3, valid_dev_lex3, inverse_vocab, overlap_map_for_fw)
