from preprocessing_utils import convert2indices, preprocess_nlg_text
from nltk.tokenize.moses import MosesTokenizer
from collections import defaultdict
from data_loaders.nlg_utils import *
from data_loaders.lex_features_utils import *

name_tok = 'XNAMEX'
near_tok = 'XNEARX'
food_tok = 'XFOODX'

tokenizer = MosesTokenizer()


def _load_nlg_data(fname):
    data_reader = csv.DictReader(open(fname, encoding='utf-8', mode='rt'))
    processed_fields = defaultdict(lambda: [])
    outputs_raw = []
    weights_raw = []
    mr_list = []
    for row in data_reader:
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

    return outputs_raw, processed_fields, weights_raw


def load_text_gen_data(fname, feature_fname, tree_fname, config_data, vocabulary, noutputs=3, random_output=False, word_based=False, random_first_word=False):
    max_output_length = config_data['max_sentence_len']
    feature_list = config_data['features']
    dummy_word_idx = len(vocabulary)

    if word_based:
        vocabulary = {token: idx for token, (idx, freq) in vocabulary.items()}

    field_ops = {
        'eatType': 3,
        'priceRange': 6,
        'customer rating': 6,
        'food': 7,
        'area': 2,
        'familyFriendly': 2
    }

    inputs = []
    outputs_raw, processed_fields, weights_raw = _load_nlg_data(fname)

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
    target_idx = convert2indices(outputs_delex, vocabulary, dummy_word_idx, dummy_word_idx, max_sent_length=max_output_length)
    if not random_first_word:
        nsentence_embeddings, tr_fwords_full_vectors, tr_fphrase_full_vectors, tr_fpos_full_vectors, tr_fwords_vectors, tr_fpos_vectors, tr_fphrase_vectors = load_lex_features(feature_fname, config_data)
        pos_tag_feature, phrase_tag_feature = load_special_tags(tree_fname, config_data)
    else:
        nsentence_embeddings, tr_fwords_full_vectors, tr_fphrase_full_vectors, tr_fpos_full_vectors, tr_fwords_vectors, tr_fpos_vectors, tr_fphrase_vectors = sample_lex_features(processed_fields, config_data)
        pos_tag_feature, phrase_tag_feature = sample_special_tags(processed_fields, config_data)

    if 'nsent' in feature_list:
        inputs.append(nsentence_embeddings)

    if 'fout_word_vectors':
        inputs.append(tr_fwords_full_vectors)

    if 'fout_phrase_vectors':
        inputs.append(tr_fphrase_full_vectors)

    if 'fout_pos_vectors':
        inputs.append(tr_fpos_full_vectors)

    if 'fword_vectors':
        inputs.extend(tr_fwords_vectors)

    if 'fphrase_vectors':
        inputs.extend(tr_fphrase_vectors)

    if 'fpos_vectors':
        inputs.extend(tr_fpos_vectors)

    if 'pos_tag_feature':
        inputs.append(pos_tag_feature)

    if 'phrase_tag_feature':
        inputs.append(phrase_tag_feature)

    if random_output:
        target_idx = np.random.normal(loc=0, scale=0.25, size=target_idx.shape)
    inputs.append(target_idx)
    weights = np.array(weights_raw)

    outputs = [np.ones(len(inputs[0]))] * noutputs

    lex_dict = {
        name_tok: processed_fields['name'],
        near_tok: processed_fields['near'],
        food_tok: processed_fields['food'],
    }

    return inputs, outputs, [weights]*noutputs, lex_dict


if __name__ == '__main__':
    config_data = json.load(open('configurations/config_vae_nlg.json'))
    tweets_path = config_data['tweets_path']

    vocab_path = config_data['vocab_path']
    vocab = cPickle.load(open(join(vocab_path, 'vocabulary.pkl'), 'rb'))

    inputs, _, _, _ = load_text_gen_data(join(tweets_path, 'devset.csv'), join(tweets_path, 'devset_lex_features.csv'), config_data, vocab, random_first_word=True)
