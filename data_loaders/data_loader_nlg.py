from preprocessing_utils import convert2indices, preprocess_nlg_text
from nltk.tokenize.moses import MosesTokenizer
import csv
import numpy as np
from collections import defaultdict, Counter
import json
from os.path import join
import _pickle as cPickle
import random

name_tok = 'XNAMEX'
near_tok = 'XNEARX'
food_tok = 'XFOODX'

tokenizer = MosesTokenizer()


def load_text_gen_data(fname, config_data, vocabulary, noutputs=3, random_output=False, word_based=False, random_first_word=False):
    max_output_length = config_data['max_sentence_len']
    vocab_path = config_data['vocab_path']
    fw_vocab = cPickle.load(open(join(vocab_path, 'fw_vocab.pkl'), 'rb'))
    overlap_map_for_fw = cPickle.load(open(join(vocab_path, 'overlap_map_for_fw.pkl'), 'rb'))

    dummy_word_idx = len(vocabulary)
    dropout_word_idx = len(vocabulary) + 1
    reader = csv.DictReader(open(fname, encoding='utf-8', mode='rt'))
    if word_based:
        vocabulary = {token: idx for token, (idx, freq) in vocabulary.items()}

    headers = [
        ('name', process_name),
        ('eatType', process_eat_type),
        ('priceRange', process_price_range),
        ('customer rating', process_customer_rating),
        ('near', process_near),
        ('food', process_food),
        ('area', process_area),
        ('familyFriendly', process_family_friendly)
    ]

    field_ops = {
        'eatType': 3,
        'priceRange': 6,
        'customer rating': 6,
        'food': 7,
        'area': 2,
        'familyFriendly': 2
    }

    processed_fields = defaultdict(lambda: [])
    outputs_raw = []
    weights_raw = []
    mr_list = []
    for row in reader:
        i1 = row['mr']
        i2 = row.get('ref', '')
        i3 = row.get('weight', 1.0)
        mr_list.append(i1)
        weights_raw.append(float(i3))
        outputs_raw.append(i2)
        keywords = i1.split(',')

        kv = {}
        for keyword in keywords:
            kidx = keyword.find('[')
            key = keyword[:kidx].strip()
            value = keyword[kidx + 1: keyword.find(']')]
            kv[key] = value

        for header, funct in headers:
            val = kv.get(header, None)
            processed_value = funct(val)
            processed_fields[header].append(processed_value)

    inputs = []

    for header, _ in headers:
        values = processed_fields[header]
        if header in ['name', 'near', 'food']:
            value_idx = []
            for value in values:
                x = np.zeros(2)
                if value:
                    x[0] = 1
                else:
                    x[1] = 1
                value_idx.append(x)
            value_idx = np.array(value_idx).astype('float32')
        else:
            value_idx = []
            for value in values:
                x = np.zeros(field_ops[header] + 1)
                if value is not None:
                    x[value] = 1
                value_idx.append(x)

            value_idx = np.array(value_idx).astype('float32')
        inputs.append(value_idx)

    outputs_delex = [preprocess_nlg_text(x, name, near, food, name_tok, near_tok, food_tok, word_based=word_based) for x, name, near, food in zip(outputs_raw, processed_fields['name'], processed_fields['near'],  processed_fields['food'])]
    if not random_first_word:
        first_words = get_first_words(outputs_delex, fw_vocab, random_first_word)
    else:
        first_words, _ = sample_first_word(inputs, overlap_map_for_fw, fw_vocab)
    inputs.append(first_words)
    target_idx = convert2indices(outputs_delex, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_output_length)

    if random_output:
        target_idx = np.random.normal(loc=0, scale=0.25, size=target_idx.shape)#np.ones_like(target_idx)*dropout_word_idx
    inputs.append(target_idx)
    weights = np.array(weights_raw)

    outputs = [np.ones(len(inputs[0]))] * noutputs

    lex_dict = {
        name_tok: processed_fields['name'],
        near_tok: processed_fields['near'],
        food_tok: processed_fields['food'],
    }

    return inputs, outputs, [weights]*noutputs, lex_dict


def get_first_words(outputs_delex, first_word_dict, random_first_word=False):
    first_words = []
    first_word_stats = defaultdict(lambda: 0)
    for i2 in outputs_delex:
        tokens = tokenizer.tokenize(i2)
        first_words.append(tokens[0])
        first_word_stats[tokens[0]] += 1
    idx_counter = max(first_word_dict.values()) + 1
    value_idx = []
    for word in first_words:
        x = np.zeros(idx_counter + 1)
        if random_first_word:
            idx = np.random.randint(low=0, high=idx_counter)
        else:
            idx = first_word_dict.get(word, idx_counter)
        x[idx] = 1.0
        value_idx.append(x)
    value_idx = np.array(value_idx).astype('float32')
    return value_idx


def sample_first_word(inputs, overlap_map_for_fw, first_word_dict):
    precence_idx = [1, 5, 12, 19, 21, 23, 26, 29]
    value_idx = []
    idx_counter = max(first_word_dict.values()) + 1
    first_word_indices = []
    for dp in zip(*inputs):
        diffs = {}
        x = np.concatenate(dp[:8])
        vec = np.zeros(idx_counter + 1)
        for fw_index, scores_1 in overlap_map_for_fw.items():
            diff = np.mean(np.square(x - scores_1))
            for pidx in precence_idx: #if the feature must be present or absent but isn't => thorw it out
                if scores_1[pidx] == 1.0 and x[pidx] == 0.0 or scores_1[pidx] == 0.0 and x[pidx] == 1.0:
                    diff = np.inf
            diffs[fw_index] = diff
        top_fw = [x for x in sorted(diffs.items(), key=lambda x: x[1], reverse=False)[:10]]
        first_word = random.choice(top_fw)[0]
        vec[first_word] = 1.0
        first_word_indices.append(top_fw)
        value_idx.append(vec)

    value_idx = np.array(value_idx).astype('float32')
    return value_idx, first_word_indices


def get_fist_words_for_input(dp, overlap_map_for_fw):
    precence_idx = [1, 5, 12, 19, 21, 23, 26, 29]
    diffs = {}
    x = np.concatenate(dp[:8])
    for fw_index, scores_1 in overlap_map_for_fw.items():
        diff = np.mean(np.square(x - scores_1))
        for pidx in precence_idx:  # if the feature must be present or absent but isn't => thorw it out
            if scores_1[pidx] == 1.0 and x[pidx] == 0.0 or scores_1[pidx] == 0.0 and x[pidx] == 1.0:
                diff = np.inf
        diffs[fw_index] = diff
    top_fw = [x for x in sorted(diffs.items(), key=lambda x: x[1], reverse=False)[:15]]
    #first_word = random.choice(top_fw)[0]
    return top_fw


def process_name(text):
    if text:
        return text
    else:
        return ''


def process_eat_type(text):
    return {
        'coffee shop': 0,
        'pub': 1,
        'restaurant': 2,
    }.get(text, 3)


def process_area(text):
    return {
        'city centre': 0,
        'riverside': 1
    }.get(text, 2)


def process_price_range(text):
    return {
        'high': 0,
        'more than £30': 1,
        '£20-25': 2,
        'cheap': 3,
        'less than £20': 4,
        'moderate': 5
    }.get(text, 6)


def process_customer_rating(text):
    return {
        'high': 0,
        '5 out of 5': 1,
        'average': 2,
        '3 out of 5': 3,
        'low': 4,
        '1 out of 5': 5
    }.get(text, 6)


def process_near(text):
    if text:
        return text
    else:
        return ''


def process_food(text):
    if text:
        return text
    else:
        return ''
    # return {
    #     'French': 0,
    #     'English': 1,
    #     'Japanese': 2,
    #     'Indian': 3,
    #     'Italian': 4,
    #     'Fast food': 5,
    #     'Chinese': 6
    # }.get(text, 7)


def process_family_friendly(text):
    return {
        'yes': 0,
        'no': 1
    }.get(text, 2)


def get_texts(file_path):
    reader = csv.DictReader(open(file_path, encoding='utf-8', mode='rt'))

    headers = [
        ('name', process_name),
        ('eatType', process_eat_type),
        ('priceRange', process_price_range),
        ('customer rating', process_customer_rating),
        ('near', process_near),
        ('food', process_food),
        ('area', process_area),
        ('familyFriendly', process_family_friendly)
    ]

    references = []
    processed_fields = defaultdict(lambda: [])
    for row in reader:
        i1 = row['mr']
        i2 = row['ref']

        references.append(i2)
        keywords = i1.split(',')
        kv = {}
        for keyword in keywords:
            kidx = keyword.find('[')
            key = keyword[:kidx].strip()
            value = keyword[kidx + 1: keyword.find(']')]
            kv[key] = value

        for header, funct in headers:
            val = kv.get(header, None)
            processed_value = funct(val)
            processed_fields[header].append(processed_value)

    tok_refenreces = [preprocess_nlg_text(x, name, near, food, name_tok, near_tok, food_tok, word_based=False) for x, name, near, food in zip(references, processed_fields['name'], processed_fields['near'],  processed_fields['food'])]

    return tok_refenreces


if __name__ == '__main__':
    config_data = json.load(open('configurations/config_vae_nlg.json'))
    tweets_path = config_data['tweets_path']

    vocab_path = config_data['vocab_path']
    vocab = cPickle.load(open(join(vocab_path, 'vocabulary.pkl'), 'rb'))

    inputs, _, _, _ = load_text_gen_data(join(tweets_path, 'devset_reduced.csv'), config_data, vocab, random_first_word=True)
