from preprocessing_utils import convert2indices, preprocess_nlg_text
import csv
import numpy as np
from collections import defaultdict, Counter
import json
from os.path import join
import _pickle as cPickle


def load_text_gen_data(fname, config_data, vocabulary, noutputs=3, random_output=False, word_based=False):
    max_output_length = config_data['max_output_length']
    dummy_word_idx = len(vocabulary)
    reader = csv.DictReader(open(fname, encoding='utf-8', mode='rt'))

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

    for row in reader:
        i1 = row['mr']
        i2 = row['ref']

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

    name_tok = 'X-name'
    near_tok = 'X-near'

    for header, _ in headers:
        values = processed_fields[header]
        if header in ['name', 'near']:
            value_idx = []
            for value in values:
                x = np.zeros(2)
                if value:
                    x[0] = 1
                else:
                    x[1] = 1
                value_idx.append(x)
            value_idx = np.array(value_idx).astype('float32')
            #delex
            if header == 'name':
                values = [name_tok for _ in values]
            elif header == 'near':
                values = [near_tok for _ in values]

            #value_idx = convert2indices(values, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_input_length)
        else:
            value_idx = []
            for value in values:
                x = np.zeros(field_ops[header] + 1)
                if value is not None:
                    x[value] = 1
                value_idx.append(x)

            value_idx = np.array(value_idx).astype('float32')
        inputs.append(value_idx)

    outputs_delex = [preprocess_nlg_text(x, name, near, word_based=word_based) for x, name, near in zip(outputs_raw, processed_fields['name'], processed_fields['near'])]

    target_idx = convert2indices(outputs_delex, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_output_length)
    if random_output:
        target_idx = np.ones(shape=target_idx.shape)*dummy_word_idx
    inputs.append(target_idx)

    outputs = [np.ones(len(inputs[0]))] * noutputs

    lex_dict = {
        name_tok: processed_fields['name'],
        near_tok: processed_fields['near'],
    }

    return inputs, outputs, lex_dict


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
    return {
        'French': 0,
        'English': 1,
        'Japanese': 2,
        'Indian': 3,
        'Italian': 4,
        'Fast food': 5,
        'Chinese': 6
    }.get(text, 7)


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

    tok_refenreces = [preprocess_nlg_text(x, name, near) for x, name, near in zip(references, processed_fields['name'], processed_fields['near'])]
    return tok_refenreces


if __name__ == '__main__':
    config_data = json.load(open('configurations/config_vae_nlg.json'))
    tweets_path = config_data['tweets_path']

    train_texts = get_texts(join(tweets_path, 'trainset.csv'))
    dev_texts = get_texts(join(tweets_path, 'devset.csv'))

    full_tokens = Counter()
    sentence_lengths = []
    for tokens in train_texts + dev_texts:
        full_tokens.update(tokens)
        sentence_lengths.append(len(tokens))

    print(len(full_tokens))
    print(full_tokens.most_common(100))
    sentence_lengths = np.array(sentence_lengths)
    print(np.mean(sentence_lengths))
    print(np.std(sentence_lengths))

    vocabulary = {}
    for i, (token, freq) in enumerate(full_tokens.items()):
        vocabulary[token] = (i, freq)

    cPickle.dump(vocabulary, open(join(tweets_path, 'vocab_word.pkl'), 'wb'))

    vocab = {x: i for x, (i, f) in vocabulary.items()}
    inputs, outputs, lex_dict = load_text_gen_data(join(tweets_path, 'trainset.csv'), config_data, vocab, noutputs=3, random_output=False, word_based=True)

    print('Done')

