import csv
import random
import json
import _pickle as cPickle
from os.path import join
import numpy as np


def _compute_nsent_features(nsentences, max_sentences):
    nsentence_embeddings = []
    for nsentence in nsentences:
        x = np.zeros(max_sentences)
        if nsentence <= max_sentences:
            x[nsentence - 1] = 1
        else:
            x[-1] = 1
        nsentence_embeddings.append(x)
    nsentence_embeddings = np.array(nsentence_embeddings)
    return nsentence_embeddings


def _compute_first_position_features(number_of_sentences, max_sentence_len, dummy_idx1, vocab_fp1, fp_data):
    fp_vectors = []
    for nsentence, first_position in zip(number_of_sentences, fp_data):
        tmp_fwords = []
        for n in range(1, max_sentence_len):
            x = np.zeros(dummy_idx1 + 2)
            if n < nsentence:
                word = first_position[n]
                widx = vocab_fp1.get(word, dummy_idx1)
                x[widx] = 1
            else:
                x[-1] = 1
            tmp_fwords.append(x)
        fp_vectors.append(tmp_fwords)

    transposed_fp_vectors = []
    for i in range(max_sentence_len - 1):
        fwords_i = np.array([x[i] for x in fp_vectors])
        transposed_fp_vectors.append(fwords_i)

    return transposed_fp_vectors


def _compute_first_oput_features(number_of_sentences, dummy_idx0, vocab_fp0, fp_data):
    fp_vectors = []
    for nsentence, first_position in zip(number_of_sentences, fp_data):
        x = np.zeros(dummy_idx0 + 2)
        word = first_position[0]
        widx = vocab_fp0.get(word, dummy_idx0)
        x[widx] = 1
        fp_vectors.append(x)

    return np.array(fp_vectors)


def _sample_first_position_features(number_of_sentences, max_sentences, dummy_idx1):
    fp_vectors = []
    for nsentence in number_of_sentences:
        tmp_fwords = []
        for n in range(1, max_sentences):
            x = np.zeros(dummy_idx1 + 2)
            if n < nsentence:
                widx = random.randint(0, dummy_idx1)
                x[widx] = 1
            else:
                x[-1] = 1
            tmp_fwords.append(x)
        fp_vectors.append(tmp_fwords)
    transposed_fp_vectors = []
    for i in range(max_sentences - 1):
        fwords_i = np.array([x[i] for x in fp_vectors])
        transposed_fp_vectors.append(fwords_i)

    return transposed_fp_vectors


def _sample_first_output_features(number_of_sentences, dummy_idx0):
    fp_vectors = []
    for _ in number_of_sentences:
        x = np.zeros(dummy_idx0 + 2)
        widx = random.randint(0, dummy_idx0)
        x[widx] = 1
        fp_vectors.append(x)

    return np.array(fp_vectors)


def load_lex_features(fname, config_data):
    lex_feature_reader = csv.DictReader(open(fname, encoding='utf-8', mode='rt'), delimiter='\t')
    max_sentences = config_data['max_nsentences']
    vocab_path = config_data['vocab_path']
    vocab_fwords0 = cPickle.load(open(join(vocab_path, 'vocab_fwords_0.pkl'), 'rb'))
    vocab_fwords1 = cPickle.load(open(join(vocab_path, 'vocab_fwords_1.pkl'), 'rb'))
    vocab_pos_tags = cPickle.load(open(join(vocab_path, 'vocab_pos_tags.pkl'), 'rb'))
    vocab_phrase_tags = cPickle.load(open(join(vocab_path, 'vocab_phrase_tags.pkl'), 'rb'))

    dummy_fword_idx0 = len(vocab_fwords0)
    dummy_fword_idx1 = len(vocab_fwords1)
    dummy_fpostag_idx = len(vocab_pos_tags)
    dummy_fphrtag_idx = len(vocab_phrase_tags)

    nsentences = []
    fwords = []
    fpostags = []
    fphrases = []
    for row in lex_feature_reader:
        nsentence = int(row['Number Of Sentences'])
        fword = row['First Words']
        fpostag = row['First POS tags']
        fphrase = row['First Phrases']

        nsentences.append(nsentence)
        fwords.append(fword.split(','))
        fpostags.append(fpostag.split(','))
        fphrases.append(fphrase.split(','))

    nsentence_embeddings = _compute_nsent_features(nsentences, max_sentences)
    tr_fwords_vectors = _compute_first_position_features(nsentences, max_sentences, dummy_fword_idx1, vocab_fwords1, fwords)
    tr_fphrase_vectors = _compute_first_position_features(nsentences, max_sentences, dummy_fphrtag_idx, vocab_phrase_tags, fphrases)
    tr_fpos_vectors = _compute_first_position_features(nsentences, max_sentences, dummy_fpostag_idx, vocab_pos_tags, fpostags)

    tr_fwords_full_vectors = _compute_first_oput_features(nsentences, dummy_fword_idx0, vocab_fwords0, fwords)
    tr_fphrase_full_vectors = _compute_first_oput_features(nsentences, dummy_fphrtag_idx, vocab_phrase_tags, fphrases)
    tr_fpos_full_vectors = _compute_first_oput_features(nsentences, dummy_fpostag_idx, vocab_pos_tags, fpostags)

    return nsentence_embeddings, tr_fwords_full_vectors, tr_fphrase_full_vectors, tr_fpos_full_vectors, tr_fwords_vectors, tr_fpos_vectors, tr_fphrase_vectors


def sample_lex_features(processed_fields, config_data):
    max_sentences = config_data['max_nsentences']
    vocab_path = config_data['vocab_path']
    vocab_fwords0 = cPickle.load(open(join(vocab_path, 'vocab_fwords_0.pkl'), 'rb'))
    vocab_fwords1 = cPickle.load(open(join(vocab_path, 'vocab_fwords_1.pkl'), 'rb'))
    vocab_pos_tags = cPickle.load(open(join(vocab_path, 'vocab_pos_tags.pkl'), 'rb'))
    vocab_phrase_tags = cPickle.load(open(join(vocab_path, 'vocab_phrase_tags.pkl'), 'rb'))

    dummy_fword_idx0 = len(vocab_fwords0)
    dummy_fword_idx1 = len(vocab_fwords1)
    dummy_fpostag_idx = len(vocab_pos_tags)
    dummy_fphrtag_idx = len(vocab_phrase_tags)

    nsentence_embeddings = []
    nsentence_lengths = []
    for _ in processed_fields['name']:
        nsentence = random.randint(1, max_sentences)
        nsen_emb = np.lib.pad(np.ones(shape=(nsentence,)), pad_width=(0, max([max_sentences - nsentence, 0])),
                              mode='constant', constant_values=(0, 0))
        nsentence_embeddings.append(nsen_emb)
        nsentence_lengths.append(nsentence)
    nsentence_embeddings = np.array(nsentence_embeddings)

    tr_fwords_vectors = _sample_first_position_features(nsentence_lengths, max_sentences, dummy_fword_idx1)
    tr_fphrase_vectors = _sample_first_position_features(nsentence_lengths, max_sentences, dummy_fphrtag_idx)
    tr_fpos_vectors = _sample_first_position_features(nsentence_lengths, max_sentences, dummy_fpostag_idx)

    tr_fwords_full_vectors = _sample_first_output_features(nsentence_lengths, dummy_fword_idx0)
    tr_fphrase_full_vectors = _sample_first_output_features(nsentence_lengths, dummy_fphrtag_idx)
    tr_fpos_full_vectors = _sample_first_output_features(nsentence_lengths, dummy_fpostag_idx)

    return nsentence_embeddings, tr_fwords_full_vectors, tr_fphrase_full_vectors, tr_fpos_full_vectors, tr_fwords_vectors, tr_fpos_vectors, tr_fphrase_vectors


def load_special_tags(fname, config_data):
    vocab_path = config_data['vocab_path']
    tree_feature_file = open(fname, 'rt', encoding='utf-8')
    tree_features = json.load(tree_feature_file)

    phrase_vocab = json.load(open(join(vocab_path, 'phrase_vocab.json'), 'rt', encoding='utf-8'))
    pos_vocab = json.load(open(join(vocab_path, 'pos_vocab.json'), 'rt', encoding='utf-8'))

    phrase_tag_list = []
    pos_tag_list = []
    for tree_feature in tree_features:
        phrase_vec = np.zeros(len(phrase_vocab))
        pos_vec = np.zeros(len(pos_vocab))
        for sentence_feature in tree_feature:
            phrase_tags = set(sentence_feature.keys()).intersection(phrase_vocab.keys())
            pos_tags = set(sentence_feature.keys()).intersection(pos_vocab.keys())

            for phrase_tag in phrase_tags:
                pidx = phrase_vocab[phrase_tag]
                phrase_vec[pidx] = 1

            for pos_tag in pos_tags:
                pidx = pos_vocab[pos_tag]
                pos_vec[pidx] = 1

        phrase_tag_list.append(phrase_vec)
        pos_tag_list.append(pos_vec)

    phrase_tag_feature = np.array(phrase_tag_list)
    pos_tag_feature = np.array(pos_tag_list)

    return pos_tag_feature, phrase_tag_feature


def sample_special_tags(processed_fields, config_data):
    vocab_path = config_data['vocab_path']

    phrase_vocab = json.load(open(join(vocab_path, 'phrase_vocab.json'), 'rt', encoding='utf-8'))
    pos_vocab = json.load(open(join(vocab_path, 'pos_vocab.json'), 'rt', encoding='utf-8'))

    phrase_tag_list = []
    pos_tag_list = []

    for _ in processed_fields['name']:
        nphrase_tags = random.randint(2, 7)
        npostages = random.randint(4, 9)

        phrase_tags = random.sample(phrase_vocab.keys(), nphrase_tags)
        pos_tags = random.sample(pos_vocab.keys(), npostages)

        phrase_vec = np.zeros(len(phrase_vocab))
        pos_vec = np.zeros(len(pos_vocab))

        for phrase_tag in phrase_tags:
            pidx = phrase_vocab[phrase_tag]
            phrase_vec[pidx] = 1

        for pos_tag in pos_tags:
            pidx = pos_vocab[pos_tag]
            pos_vec[pidx] = 1

        phrase_tag_list.append(phrase_vec)
        pos_tag_list.append(pos_vec)

    phrase_tag_feature = np.array(phrase_tag_list)
    pos_tag_feature = np.array(pos_tag_list)

    return pos_tag_feature, phrase_tag_feature


def get_lengths(config_data):
    vocab_path = config_data['vocab_path']
    max_sentences = config_data['max_nsentences']

    vocab_fwords0 = cPickle.load(open(join(vocab_path, 'vocab_fwords_0.pkl'), 'rb'))
    vocab_fwords1 = cPickle.load(open(join(vocab_path, 'vocab_fwords_1.pkl'), 'rb'))
    vocab_pos_tags = cPickle.load(open(join(vocab_path, 'vocab_pos_tags.pkl'), 'rb'))
    vocab_phrase_tags = cPickle.load(open(join(vocab_path, 'vocab_phrase_tags.pkl'), 'rb'))

    dummy_fword_idx0 = len(vocab_fwords0)
    dummy_fword_idx1 = len(vocab_fwords1)
    dummy_fpostag_idx = len(vocab_pos_tags)
    dummy_fphrtag_idx = len(vocab_phrase_tags)

    phrase_vocab = json.load(open(join(vocab_path, 'phrase_vocab.json'), 'rt', encoding='utf-8'))
    pos_vocab = json.load(open(join(vocab_path, 'pos_vocab.json'), 'rt', encoding='utf-8'))

    lens = {
        'nsent': max_sentences,
        'fout_word_vectors': dummy_fword_idx0 + 2,
        'fout_phrase_vectors': dummy_fphrtag_idx + 2,
        'fout_pos_vectors': dummy_fpostag_idx + 2,
        'fword_vectors': dummy_fword_idx1 + 2,
        'fphrase_vectors': dummy_fphrtag_idx + 2,
        'fpos_vectors': dummy_fpostag_idx + 2,
        'pos_tag_feature': len(pos_vocab),
        'phrase_tag_feature': len(phrase_vocab),
    }

    return lens


def get_number_outputs(config_data):
    features = config_data['features']
    max_sentences = config_data['max_nsentences']

    lens = {
        'nsent': 1,
        'fout_word_vectors': 1,
        'fout_phrase_vectors': 1,
        'fout_pos_vectors': 1,
        'fword_vectors': max_sentences - 1,
        'fphrase_vectors': max_sentences - 1,
        'fpos_vectors': max_sentences - 1,
        'pos_tag_feature': 1,
        'phrase_tag_feature': 1,
    }

    sum = 0
    for feature in features:
        sum += lens[feature]

    return sum