import json
from os.path import join
from data_loaders.data_loader_nlg import load_text_gen_data
import pprint
import numpy as np
import _pickle as cPickle
from collections import defaultdict
from tqdm import tqdm


pp = pprint.PrettyPrinter(indent=2)

headers = [
    'name',
    'eatType',
    'priceRange',
    'customer rating',
    'near',
    'food',
    'area',
    'familyFriendly',
]


def generate_value_data(data):
    X, Y = [], []
    for dp in zip(*data):
        x = np.concatenate(dp[:8])
        y = dp[8]
        X.append(x)
        Y.append(np.argmax(y))

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def generate_feauture_data(data):
    X, Y = [], []
    for dp in zip(*data):
        ft_x = []
        for d in dp[:8]:
            x = 1 - d[-1]
            ft_x.append(x)

        x = np.array(ft_x)
        y = dp[8]
        X.append(x)
        Y.append(np.argmax(y))

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

if __name__ == '__main__':
    config_data = json.load(open('configurations/config_vae_nlg.json'))
    tweets_path = config_data['tweets_path']

    vocab_path = config_data['vocab_path']
    vocab = cPickle.load(open(join(vocab_path, 'vocabulary.pkl'), 'rb'))
    fw_vocab = cPickle.load(open(join(vocab_path, 'fw_vocab.pkl'), 'rb'))

    fw_inverse = {v: k for k,v in fw_vocab.items()}

    inputs, _, _, _ = load_text_gen_data(join(tweets_path, 'trainset.csv'), config_data, vocab)
    val_inputs, _, _, _ = load_text_gen_data(join(tweets_path, 'devset_reduced.csv'), config_data, vocab)

    X_train, y_train = generate_value_data(inputs)
    X_ft_train, y_ft_train = generate_feauture_data(inputs)
    X_val, y_val = generate_value_data(val_inputs)
    X_ft_val, y_ft_val = generate_feauture_data(val_inputs)

    possible_first_words = defaultdict(lambda: defaultdict(lambda: 0))
    feature_for_fw = defaultdict(lambda: [])
    feature_count = defaultdict(lambda: defaultdict(lambda: 0))
    for x, y in tqdm(zip(X_train, y_train)):
        s = ''.join([str(int(i)) for i in x])
        possible_first_words[s][y] += 1
        feature_for_fw[y].append(s)
        for i, val in enumerate(s):
            feature_count[i][val] += 1

    overlap_map_for_fw = {}
    for y, feature_vec_list in feature_for_fw.items():
        word = fw_inverse.get(y, 'DUMMY')
        ycount = defaultdict(lambda: defaultdict(lambda: 0))
        scores_0 = []
        scores_1 = []
        for s in feature_vec_list:
            for i, val in enumerate(s):
                ycount[i][val] += 1

        for yi in ycount.keys():
            count_0 = ycount[yi]['0']
            count_1 = ycount[yi]['1'] #how often the feature == 1 for fw y

            total_count_0 = feature_count[yi]['0']
            total_count_1 = feature_count[yi]['1']

            score_0 = count_0/(len(feature_vec_list)) if (len(feature_vec_list)) > 0 else 0 #compute confidence
            score_1 = count_1/(len(feature_vec_list)) if (len(feature_vec_list)) > 0 else 0

            scores_0.append(score_0)
            scores_1.append(score_1)

        #print(word)
        #print(['{0}: {1:0.4f}'.format(i, val) for i, val in enumerate(scores_0)])
        #print(['{0}: {1:0.4f}'.format(i, val) for i, val in enumerate(scores_1)])
        overlap_map_for_fw[y] = np.asarray(scores_1)

    cPickle.dump(overlap_map_for_fw, open('en_full/overlap_map_for_fw.pkl', 'wb'))
    precence_idx = [1, 5, 12, 19, 21, 23, 26, 29]

    ofile = open('output.txt', 'wt', encoding='utf-8')
    for i, (x, y) in enumerate(zip(X_val, y_val), start=2):
        s = ''.join([str(int(i)) for i in x])
        diffs = {}
        for fw_index, scores_1 in overlap_map_for_fw.items():
            diff = np.mean(np.square(x - scores_1))
            for pidx in precence_idx: #if the feature must be present or absent but isn't => thorw it out
                if scores_1[pidx] == 1.0 and x[pidx] == 0.0 or scores_1[pidx] == 0.0 and x[pidx] == 1.0:
                    diff = np.inf

            diffs[fw_index] = diff

        oline = '{} {}\n'.format(i, ['{}, {}'.format(fw_inverse.get(w, 'DUMMY'), s) for w, s in sorted(diffs.items(), key=lambda x: x[1], reverse=False)])
        ofile.write(oline)