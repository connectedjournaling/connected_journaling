import cluster_important_words as cluster
import sentiment_classifier as sentiment
from pathlib import Path
import helper_functions as help_fun
import numpy as np
from NER import NER
import re

## CONSTANTS ##
storage_location = Path("C:/Users/hsuen/Desktop/connected_journaling/connected_journaling/data/bbc-text.csv")
pre_trained_embeddings_path = Path('C:/Users/hsuen/Desktop/bigData/GoogleNews-vectors-negative300.bin')
sentiment_network_path = 'C:/Users/hsuen/Desktop/connected_journaling/connected_journaling/Sentiment_functions/models/bad_model.h5'

embedding_size_constant = 300
sequence_length = 1850




## GET THE IMPORTANT WORDS ##
print('CLUSTERING IMPORTANT WORDS')
# returns a two column data frame, "word", "group".
imp_word_list, num_groups = cluster.get_important_words(storage_location, pre_trained_embeddings_path)

# convert the data to be all sentences #
sentences = help_fun.split_into_sentence(storage_location)

## GRAB THE SENTIMENT & CATEGORY FOR EACH WORD ##

# load trained model #
print('LOADING TRAINED SENTIMENT CLASSIFIER')
model = sentiment.load_model(
    sentiment_network_path,
    sequence_length, embedding_size_constant)

print('CONVERTING TEST DATA')
# prepare data to be used by the model
test_data = sentiment.convert_data_test(sentences, sequence_length, pre_trained_embeddings_path,
                                        embedding_size_constant)

# loop through each sentences looking for the words in the important words list
# add columns to sum up the amount of "positive"-ness
pos_sentiment = np.zeros(len(imp_word_list))
neg_sentiment = np.zeros(len(imp_word_list))
imp_word_list.columns = ['words', 'group']

imp_word_list['pos_sentiment'] = pos_sentiment
imp_word_list['neg_sentiment'] = neg_sentiment

# initialize an NER
NER_classifier = NER()
word_model = "en_core_web_md"



# create a list of lists for the number of groups we have
categories_list = []
for x in range(num_groups):
    categories_list.append([])

list_imp_words = list(imp_word_list['words'])

print('Finding sentiment and associated categories....')
for idx, sentence in enumerate(sentences):
    token_sentence = re.compile('\w+').findall(sentence) # extract words from the sentence
    for word in token_sentence:
        if word in list_imp_words:
            print('Important word found!')
            result = model.predict(np.array(test_data[idx][:][:], ndmin=3))
            if result > 0.5:
                imp_word_list.loc[list_imp_words.index(word), 'pos_sentiment'] += result[0]

            else:
                result = (1 - result)
                imp_word_list.loc[list_imp_words.index(word), 'neg_sentiment'] += result[0]

            # now we need to determine what groups it is a part of
            entities = NER_classifier.evaluate([sentence], word_model)
            for idx2, entity in enumerate(entities[0]):
                group_num = imp_word_list.loc[list_imp_words.index(word), 'group']

                group_num = group_num - 1 # get the indexing to start @ 0
                categories_list[group_num].append(entity[1])

print('here')


