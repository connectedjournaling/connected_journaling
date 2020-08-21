import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from numpy import savetxt
import string

# pip install tweet-preprocessor
import preprocessor as p

air_path = '/Users/petergramaglia/Documents/GitHub/new_connected/connected_journaling/data/Airline_tweets.csv'
gop_path = '/Users/petergramaglia/Documents/GitHub/new_connected/connected_journaling/data/GOP_debate_tweets.csv'
tweet_pos_path = '/Users/petergramaglia/Documents/GitHub/new_connected/connected_journaling/data/nltk_twitter_samples/positive_tweets.json'
tweet_neg_path = '/Users/petergramaglia/Documents/GitHub/new_connected/connected_journaling/data/nltk_twitter_samples/negative_tweets.json'

air = (pd.read_csv(air_path)).to_numpy()
gop = (pd.read_csv(gop_path)).to_numpy()
pos = (pd.read_json(tweet_pos_path, lines=True)).to_numpy()
neg = (pd.read_json(tweet_neg_path, lines=True)).to_numpy()

# print(air[0])
# print(air[0][1])
# print(air[0][10])

cutoff = 0.75

air_clean = [[0,0]]
for i in range(0,len(air)):
    #print(air[i][2])
    if air[i][2] > cutoff:
        if air[i][1] == 'negative':
            air_clean.append([air[i][10],'positive'])
        elif air[i][1] == 'positive':
            air_clean.append([air[i][10],'positive'])

air_clean = np.delete(air_clean, 0, 0)
# print("---------------------------------------------------------------")
# print(air_clean[0:3])
print("Airlines data is clean!")

# print(gop[0])
# print(gop[0][5])
# print(gop[0][6])

gop_clean = [[0,0]]
for i in range(0,len(gop)):
    # print(gop[i][6])
    if gop[i][6] > cutoff:
        if gop[i][5] == 'Negative':
            gop_clean.append([gop[i][15], 'negative'])
        elif gop[i][5] == 'Positive':
            gop_clean.append([gop[i][15],'positive'])

gop_clean = np.delete(gop_clean, 0, 0)
# print("---------------------------------------------------------------")
# print(gop_clean[0:3])
print("GOP data is clean!")

pos_clean = [[0,0]]
for i in range(0,len(pos)):
    pos_clean.append([pos[i][2], 'positive'])

pos_clean = np.delete(pos_clean, 0, 0)
print("Pos data is clean!")


neg_clean = [[0,0]]
for i in range(0, len(neg)):
    neg_clean.append([neg[i][2], 'negative'])

neg_clean = np.delete(neg_clean, 0, 0)
print("Neg data is clean!")

all_tweets_clean = np.concatenate((air_clean, gop_clean, pos_clean, neg_clean), axis=0)


for i in range(len(all_tweets_clean)):
    phrase = all_tweets_clean[i][0]
    phrase = phrase.replace(',', '')
    phrase = phrase.replace('"', '')
    phrase = phrase.replace("'", '')
    phrase = phrase.replace("`", '')
    all_tweets_clean[i][0] = p.clean(phrase)
#
# for i in range(1,20):
#     print("------------------------------------------------------------------------")
#     phrase = all_tweets_clean[i][0]
#     print("1: ", phrase)
#     phrase = phrase.replace(',', '')
#     print(phrase)
#     phrase = p.clean(phrase)
#     print("2: ", phrase)
#     all_tweets_clean[i][1] = phrase

shuffled_tweets = shuffle(all_tweets_clean)
print(shuffled_tweets[0:30])

savetxt('/Users/petergramaglia/Documents/GitHub/new_connected/connected_journaling/data/tweets_shuffled.csv', shuffled_tweets, delimiter=',', fmt='%s')
print("Saved to tweets_shuffled.csv!")
