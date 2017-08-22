from preprocessing_utils import convert2indices
import csv
import numpy as np
from collections import defaultdict
import json
from os.path import join
import _pickle as cPickle


def load_text_gen_data(fname, config_data, vocabulary, noutputs=3):
    max_input_length = config_data['max_input_length']
    max_output_length = config_data['max_output_length']
    max_idx = max(vocabulary.values())
    dummy_word_idx = max_idx + 1
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
    for header, _ in headers:
        values = processed_fields[header]
        if header in ['name', 'near']:
            value_idx = convert2indices(values, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_input_length)
        else:
            value_idx = []
            for value in values:
                x = np.zeros(field_ops[header])
                if value:
                    x[value] = 1
                value_idx.append(x)

            value_idx = np.array(value_idx).astype('float32')
        inputs.append(value_idx)

    target_idx = convert2indices(outputs_raw, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_output_length)
    inputs.append(target_idx)

    outputs = [np.ones(len(inputs[0]))] * noutputs

    return inputs, outputs


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
    }.get(text, None)


def process_area(text):
    return {
        'city centre': 0,
        'riverside': 1
    }.get(text, None)


def process_price_range(text):
    return {
        'high': 0,
        'more than £30': 1,
        '£20-25': 2,
        'cheap': 3,
        'less than £20': 4,
        'moderate': 5
    }.get(text, None)


def process_customer_rating(text):
    return {
        'high': 0,
        '5 out of 5': 1,
        'average': 2,
        '3 out of 5': 3,
        'low': 4,
        '1 out of 5': 5
    }.get(text, None)


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
    }.get(text, None)


def process_family_friendly(text):
    return {
        'yes': 0,
        'no' : 1
    }.get(text, None)


if __name__ == '__main__':
    config_data = json.load(open('configurations/config_vae.json'))
    vocab_char_path = join(config_data['vocab_path'], 'vocabulary.pkl')

    vocab_char = cPickle.load(open(vocab_char_path, 'rb'))
    tweets_fname = join(config_data['tweets_path'], 'trainset.csv')
    i, o = load_text_gen_data(tweets_fname, config_data, vocab_char, noutputs=2)
    pass