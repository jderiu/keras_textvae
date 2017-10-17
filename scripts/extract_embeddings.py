import os
import _pickle as cPickle
import numpy as np
from tqdm import tqdm


base_path = 'en_full'

vocab_path = os.path.join(base_path, 'vocab_word.pkl')
emb_path = os.path.join('D:/GitRepos', 'wiki.en.vec')

embeddings_txt = open(emb_path, 'rt', encoding='utf-8')
emb_dims = embeddings_txt.readline().split(' ')

nvocab = int(emb_dims[0])
emb_dims = int(emb_dims[1])

print(nvocab)
print(emb_dims)

vocabulary = cPickle.load(open(vocab_path, 'rb'))
ntokens = max([idx for idx, y in vocabulary.values()])
dummy_word_idx = ntokens + 1
dropout_word_idx = ntokens + 2

scale = np.sqrt(2/(nvocab + emb_dims))
embedding_matrix = np.random.normal(loc=0.0, scale=scale, size=(dropout_word_idx + 1, emb_dims))
known_tokens = []
for embedding_line in tqdm(embeddings_txt, total=nvocab):
    emb_line = embedding_line.split()
    token = ' '.join(emb_line[:len(emb_line) - emb_dims])
    vector = np.asarray([float(x) for x in emb_line[-emb_dims:]])

    vidx, _ = vocabulary.get(token, (None, None))

    if vidx is not None:
        embedding_matrix[vidx, :] = vector
        known_tokens.append(token)

unknown_tokens = set(vocabulary.keys()).difference(known_tokens)
print(len(unknown_tokens))
print(unknown_tokens)
for utok in unknown_tokens:
    if utok.lower() in known_tokens:
        vidx, _ = vocabulary.get(utok, (None, None))
        kn_idx, _ = vocabulary.get(utok.lower(), (None, None))
        if vidx and kn_idx:
            embedding_matrix[vidx, :] = embedding_matrix[kn_idx, :]
            known_tokens.append(utok)

unknown_tokens = set(vocabulary.keys()).difference(known_tokens)
print(len(unknown_tokens))
print(unknown_tokens)
print(len(known_tokens))
embedding_matrix[dropout_word_idx, :] = np.zeros(shape=(emb_dims, ))
print(embedding_matrix[dropout_word_idx, :])
np.save(os.path.join(base_path, 'wiki_embeddings.npy'), embedding_matrix)
