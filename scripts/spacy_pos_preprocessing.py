import spacy
from os.path import join
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

nlp = spacy.load('de_core_news_sm', disable=['ner', 'parser', 'textcat'])
print(nlp.pipe)
wiki_path = 'F:/Wikipedia Embeddings/'

ifile = open(join(wiki_path, 'wiki_sentences_0.de.txt'), 'rt', encoding='utf-8')
ofile = open(join(wiki_path, 'wiki_pos_sentences_0.de.txt'), 'wt', encoding='utf-8')

batch_size = 8192
current_batch = []

for line in tqdm(ifile, total=83146114):
    current_batch.append(line.replace('\n', '').replace('\r', '').replace('\t', ''))

    if len(current_batch) >= batch_size:
        for doc in nlp.pipe(current_batch, n_threads=4, batch_size=2048):
            pos_tags = []
            for token in doc:
                pos_tag = token.pos_
                pos_tags.append(pos_tag)
            ofile.write(' '.join(pos_tags) + '\n')
        current_batch = []
