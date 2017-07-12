from sklearn.cluster import KMeans
import numpy as np
import os
import json

config_fname = os.path.join('../configurations', 'config_vae_word.json')
json_data = open(config_fname, 'r')
config_data = json.load(json_data)

vocab_path = config_data['vocab_path']
embedding_matrix = np.load(open(os.path.join(vocab_path, 'embedding_matrix.npy'), 'rb'))

print(embedding_matrix.shape)

kmeans = KMeans(n_clusters=100, random_state=0)