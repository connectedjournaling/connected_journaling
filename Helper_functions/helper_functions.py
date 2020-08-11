import nltk
import pandas as pd

import numpy as np
import string
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from pathlib import Path
import copy

table = str.maketrans('', '', string.punctuation)


def shuff_split(dataset):
    print("Shuffling/ splitting...")
    # Shuffle, train/test split
    shuffled_dataset = shuffle(dataset)
    train_dataset, test_dataset = train_test_split(
        shuffled_dataset, test_size=0.5, random_state=1)

    print("Dividing...............")
    # Divide up x and y
    train_x = train_dataset.iloc[:, 0]
    train_y = train_dataset.iloc[:, 1]
    test_x = test_dataset.iloc[:, 0]
    test_y = test_dataset.iloc[:, 1]

    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    # Returns numpy arrays
    return train_x, train_y, test_x, test_y


def remove_br(sentences):
    new_sentences = copy.deepcopy(sentences)
    for i in range(0, len(new_sentences)):
        new_sentences[i] = new_sentences[i].replace('<br />', '')
    return new_sentences


def just_words(sentences):
    new_sentences = copy.deepcopy(sentences)
    for i in range(0, len(new_sentences)):
        new_sentences[i] = new_sentences[i].split()
        new_sentences[i] = [word.translate(table) for word in new_sentences[i]]
        new_sentences[i] = [word.lower() for word in new_sentences[i]]
        new_sentences[i] = [word for word in new_sentences[i] if word not in stopwords.words('english')]
    return new_sentences


def convert_y(input):
    print("Y stuff")
    new_input = np.zeros(len(input))

    for i in range(0, len(input)):
        if input[i] == "positive":
            new_input[i] = 1
    return new_input


# Takes in Pandas Dataframe
def pre_processing(dataset):
    # These returned values below are numpy arrays. x is an array where each
    # element is a string containing the review text (length is num reviews,
    # width is just one element). y is an array where each element is either
    # 'positive' or 'negative'.
    train_x, train_y, test_x, test_y = shuff_split(dataset)

    train_x = remove_br(train_x)
    test_x = remove_br(test_x)

    train_x = just_words(train_x)
    test_x = just_words(test_x)

    train_y = convert_y(train_y)
    test_y = convert_y(test_y)

    return train_x, train_y, test_x, test_y


def prepare_new_predictions(sentences):
    new_sentences = remove_br(sentences)
    new_sentences = just_words(new_sentences)
    return new_sentences


def split_into_sentence(documents):
    sentences = []
    documents = pd.read_csv(documents)
    documents = documents['text']
    for document in documents:
        sent_text = nltk.sent_tokenize(document)
        for sentence in sent_text:
            sentences.append(sentence)

    return sentences
