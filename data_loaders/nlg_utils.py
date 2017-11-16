import numpy as np
import random

def process_family_friendly(text):
    return {
        'yes': 0,
        'no': 1
    }.get(text, 2)


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