import csv
import spacy
import math
from tqdm import tqdm
from collections import defaultdict
import itertools

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parse', 'tagger', 'textcat'])

name_tok = 'XNAMEX'
near_tok = 'XNEARX'
food_tok = 'XFOODX'

headers = ['name', 'eatType', 'priceRange', 'customer rating', 'near', 'food', 'area', 'familyFriendly']

trainset_reader = csv.DictReader(open('en_full/trainset.csv', encoding='utf-8', mode='rt'))

feature_list = []
sentences = []
raw_mr_list = []
for row in tqdm(trainset_reader):
    i1 = row['mr']
    i2 = row['ref']

    kv = {}
    keywords = i1.split(',')
    for keyword in keywords:
        kidx = keyword.find('[')
        key = keyword[:kidx].strip()
        value = keyword[kidx + 1: keyword.find(']')]
        kv[key] = value

    for header, reptok in zip(['name', 'near'], [name_tok, near_tok]):
        if kv.get(header, None) is not None:
            i2 = i2.replace(kv.get(header, None), reptok)
            i1 = i1.replace(kv.get(header, None), reptok)
            kv[header] = reptok

    feature_list.append(kv)
    sentences.append(i2)
    raw_mr_list.append(i1)

doc_sentences = []
doc_vectors = []

for doc in tqdm(nlp.pipe(sentences, n_threads=4, batch_size=2048)):
    doc_sentences.append(doc)

vec_for_kv = defaultdict(lambda: [])
for kv_raw, vec in zip(raw_mr_list, doc_sentences):
    vec_for_kv[kv_raw].append(vec)

avg_for_kv = {}
for i, (kv_raw, docs) in enumerate(vec_for_kv.items()):
    scores = []
    for d1, d2 in itertools.product(docs, repeat=2):
        score = d1.similarity(d2)
        scores.append(score)
    avg_for_kv[kv_raw] = sum(scores)/len(scores)
    print(i, kv_raw, avg_for_kv[kv_raw])

n_total = 0
tf_token_total = defaultdict(lambda: 0)
key_value_to_sentence = defaultdict(lambda: defaultdict(lambda: []))
for kv, doc in tqdm(zip(feature_list, doc_sentences)):
    for key, value in kv.items():
        key_value_to_sentence[key][value].append(doc)
        for token in doc:
            tf_token_total[token.text] += 1
            n_total += 1

tf_key_values = defaultdict(lambda: {})
idf_token = defaultdict(lambda: defaultdict(lambda: 0))
n_value = defaultdict(lambda: {})
for key, value_to_sentence in key_value_to_sentence.items():
    for value, sen_for_val in value_to_sentence.items():
        tmp_tf_value = defaultdict(lambda: 0)
        for sentence in sen_for_val:
            for token in sentence:
                tmp_tf_value[token.text] += 1

        n_value[key][value] = sum(tmp_tf_value.values())
        tf_key_values[key][value] = tmp_tf_value
        for token in tmp_tf_value.keys():
            idf_token[key][token] += 1

tf_idf_key_val = defaultdict(lambda: defaultdict(lambda: {}))
for key, tf_values in tf_key_values.items():
    for value in tf_values.keys():
        for token, tf_score in tf_values[value].items():
            df_score = idf_token[key][token]

            idf = len(tf_values) / df_score
            log_idf = math.log(1 + idf)

            tfidf_unormalized = tf_score * log_idf
            tfidf_log_weighted = (1 + math.log(tf_score)) * log_idf

            val_ratio = tf_score / n_value[key][value]
            tot_ratio = tf_token_total[token] / n_total
            lockword_score = val_ratio / tot_ratio

            if tf_score > 50 and lockword_score > 3:
                tf_idf_key_val[key][value][token] = (tfidf_log_weighted, lockword_score, tf_score)

for key, tf_idf in tf_idf_key_val.items():
    print(key)
    for value, tfidf_values in tf_idf.items():
        print(value)
        for token, score in sorted(tfidf_values.items(), key=lambda x: x[1][0]*x[1][1], reverse=True)[:15]:
            print(token, score)
        print()