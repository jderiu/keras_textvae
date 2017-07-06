import os

tweets_path = '../en_full'

full_tweets = []
for fname in os.listdir(tweets_path):
    if fname.endswith('.tsv'):
        file_path = os.path.join(tweets_path, fname)
        idata = open(file_path, 'rt', encoding='utf-8', newline='\n').readlines()
        itweets = [x.split('\t')[-1] for x in idata]
        full_tweets.extend(itweets)

ofile = open(os.path.join(tweets_path, 'tweets.txt'), 'wt', encoding='utf-8')
ofile.writelines(full_tweets)
ofile.close()