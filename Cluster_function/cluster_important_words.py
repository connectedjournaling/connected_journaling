from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as lda
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from spacy.lang.en.stop_words import STOP_WORDS

# constants defined here:
additional_stop_words = ['said']

# cutoff = 0.7 and vocab = 100 are a good trade-off
TFID_cutoff = 0.7
vocab_size = 200
clustering_algo = 'kmeans'  # options are 'kmeans' & 'DBSCAN'

# cluster used in both kmeans and LDA
num_clusters = 5

# 2,2 is pretty good
# params for DBSCAN:
eps_value = 3
minimum_samples = 3

# number of words to be included for LDA
num_LDA_words = 5

# define a complete list of stopwords
STOPWORDS = STOPWORDS | STOP_WORDS
for word_elem in additional_stop_words:
    STOPWORDS.add(word_elem)



def get_important_words(train_text, word_vectorizer):
    # Parameter Options:
    # the "input" parameter can also be "filename" or "file" if we want to
    # read from a file. @ the moment we pass in the raw words
    # the "stop words" are a list of words to be ignored. Default list of "english" can be used
    # "Analyzer" means that sequences of words (or chars) are going to be the features
    # "ngrams"= (1,2) means we accept both unigrams and bigrams as potential features

    vectorizer = TfidfVectorizer(input='content', stop_words=STOPWORDS, lowercase=True, encoding='UTF-8',
                                 strip_accents='unicode', analyzer='word', ngram_range=(1, 2), max_features=vocab_size)


    # feature_matrix will be sparse, use "feature_matrix.toarray()" to get an array representation
    # all values will have been normalized to be between 0 and 1
    feature_matrix = vectorizer.fit_transform(train_text)

    # now we want to get the most important words in each document
    feature_matrix = feature_matrix.toarray()
    corpus_vocab = vectorizer.get_feature_names()
    corpus_vocab = np.array(corpus_vocab)
    imp_word_list = []
    for document in feature_matrix:
        top_words = document > TFID_cutoff
        top_words = corpus_vocab[top_words]
        for word in top_words:
            if word not in imp_word_list:
                imp_word_list.append(word)

    # by the end of this we get an array of all the important words from each document

    # load in a pre-trained word-embedding from google's word2vec
    #model = KeyedVectors.load_word2vec_format(path_to_embeddings,
                                              #binary=True)

    model = word_vectorizer

    # create a new feature matrix for each important from above
    # each row is an important word with its vector representation
    word_vectors = np.zeros((len(imp_word_list), 300))  # the word2vec model vetorizes words to 300 dimensions
    for idx, word in enumerate(imp_word_list):
        try:
            word_vector = model[word]
            word_vectors[idx, :] = word_vector
        except Exception as e:
            print(e)
            continue

    # run clustering algorithms from the top words
    if clustering_algo == 'kmeans':
        # Run k-means to get groups of similar words together
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0)
        labels = kmeans.fit_predict(word_vectors)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, num_clusters)]
        n_clusters_ = num_clusters

    elif clustering_algo == 'DBSCAN':
        db = DBSCAN(eps=eps_value, min_samples=minimum_samples).fit(word_vectors)
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]

        # make "noise" black
        colors[-1] = [0, 0, 0, 1]

    # Run PCA so we can actually visualize this in two dimensions
    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(word_vectors)

    x_points = [o[0] for o in scatter_plot_points]
    y_points = [o[1] for o in scatter_plot_points]

    plt.scatter(x_points, y_points, c=[colors[d] for d in labels])

    for i, txt in enumerate(imp_word_list):
        plt.annotate(txt, (x_points[i], y_points[i]))

    plt.title('Number of clusters: %d' % n_clusters_)
    plt.show()

    # return a dataframe of the important words and their grouping
    return pd.DataFrame([imp_word_list, labels]).T, num_clusters


def lda_cluster(train_text):
    vectorizer = TfidfVectorizer(input='content', stop_words='english', lowercase=True, encoding='UTF-8',
                                 strip_accents='unicode', analyzer='word', ngram_range=(1, 2), max_features=100)


    # feature_matrix will be sparse, use "feature_matrix.toarray()" to get an array representation
    feature_matrix = vectorizer.fit_transform(train_text)
    corpus_vocab = vectorizer.get_feature_names()
    corpus_vocab = np.array(corpus_vocab)


    # now we want to get the most important words in each document
    feature_matrix = feature_matrix.toarray()

    # Since values of feature_matrix are between zero and 1, we can say
    # that for each document, only consider the word as part of the document
    # if it is of particular importance:
    feature_matrix = feature_matrix > TFID_cutoff
    feature_matrix = feature_matrix.astype(int)

    # now feature matrix is a matrix of 1 or 0 (bag of words)
    # we use corpus_vocab to get the actual corresponding word
    my_lda = lda(n_components=num_clusters, random_state=0)

    # run the LDA algorithm
    my_lda.fit_transform(feature_matrix)

    # extract the associated probability for each word
    word_topics = my_lda.components_ # will be of size (num_topics, num_words), each cell representing probability

    top_words_list = []
    top_words_group = []

    # now that we have the top words, we want to see what the top words are in each category
    for idx in range(num_clusters):
        category = word_topics[idx][:]

        # sort the probabilities of the associated words by index
        # the below will go from smallest to largest
        sorted_words = np.argsort(category)
        top_words = sorted_words[-1*num_LDA_words:]

        print('###########')
        print('For Category number {0}'.format(idx))
        for word in top_words:
            top_words_list.append(corpus_vocab[word])
            top_words_group.append(idx)
            print(corpus_vocab[word])

    imp_word_list = pd.DataFrame(list(zip(top_words_list, top_words_group)), columns=['word', 'group'])
    return imp_word_list, num_clusters










