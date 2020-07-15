from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from spacy.lang.en.stop_words import STOP_WORDS

# constants defined here:
storage_location = "C:\\Users\\hsuen\\\Desktop\\connected_journaling\\data\\bbc-text.csv"
additional_stop_words = ['said']

# cutoff = 0.7 and vocab = 100 are a good trade-off
TFID_cutoff = 0.7
vocab_size = 100
clustering_algo = 'DBSCAN'  # options are 'kmeans' & 'DBSCAN'

# params for kmeans:
num_clusters = 5

# 2,2 is pretty good
# params for DBSCAN:
eps_value = 3
minimum_samples = 3

# define a complete list of stopwords
STOPWORDS = STOPWORDS | STOP_WORDS
for word_elem in additional_stop_words:
    STOPWORDS.add(word_elem)


def option_1():
    # Parameter Options:
    # the "input" parameter can also be "filename" or "file" if we want to
    # read from a file. @ the moment we pass in the raw words
    # the "stop words" are a list of words to be ignored. Default list of "english" can be used
    # "Analyzer" means that sequences of words (or chars) are going to be the features
    # "ngrams"= (1,2) means we accept both unigrams and bigrams as potential features

    vectorizer = TfidfVectorizer(input='content', stop_words=STOPWORDS, lowercase=True, encoding='UTF-8',
                                 strip_accents='unicode', analyzer='word', ngram_range=(1, 2), max_features=vocab_size)

    # read in the sample data
    train_text = pd.read_csv(storage_location)
    train_text = train_text['text']

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
    model = KeyedVectors.load_word2vec_format('C:\\Users\\hsuen\\Desktop\\bigData\\GoogleNews-vectors-negative300.bin',
                                              binary=True)

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
        colors = ["r", "b", "c", "y", "m"]
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

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def option_2():
    vectorizer = TfidfVectorizer(input='content', stop_words='english', lowercase=True, encoding='UTF-8',
                                 strip_accents='unicode', analyzer='word', ngram_range=(1, 2), max_features=100)

    # read in the sample data
    train_text = pd.read_csv(storage_location)
    labels = train_text['category']
    train_text = train_text['text']

    # feature_matrix will be sparse, use "feature_matrix.toarray()" to get an array representation
    feature_matrix = vectorizer.fit_transform(train_text)
    corpus_vocab = vectorizer.get_feature_names()
    corpus_vocab = np.array(corpus_vocab)

    # run k-means and cluster the documents
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
    kmean_indices = kmeans.fit_predict(feature_matrix)

    feature_matrix = feature_matrix.toarray()
    # Run PCA so we can actually visualize this in two dimensinos
    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(feature_matrix)

    colors = ["r", "b", "c", "y", "m"]

    x_points = [o[0] for o in scatter_plot_points]
    y_points = [o[1] for o in scatter_plot_points]

    color_points = [colors[d] for d in kmean_indices]

    plt.scatter(x_points[:50], y_points[:50], c=color_points[:50])

    for i, txt in enumerate(labels):
        plt.annotate(txt, (x_points[i], y_points[i]))
        if i > 30:
            break
    plt.show()

    master_word_list = {}

    # fig, ax = plt.subplots(nrows=5, ncols=1)

    for x in range(5):
        top_words_list = []
        docs = kmean_indices == x
        docs = feature_matrix[docs, :]

        for doc in docs:
            top_words = np.nonzero(doc)
            top_words = corpus_vocab[top_words]
            top_words_list = top_words_list + list(top_words)

        counter_obj = Counter(top_words_list)
        master_word_list[int(x)] = counter_obj.most_common(10)
        r = master_word_list[x]

        wordcloud = WordCloud(width=300, height=300,
                              background_color='white',
                              stopwords=STOPWORDS,
                              min_font_size=10).generate(" ".join([word for word, x in r]))
        plt.figure(figsize=(3, 3))
        ax = plt.axes()
        ax.imshow(wordcloud)
        ax.axis("off")

    # plot all the words in their respective categories
    print('here')


if __name__ == '__main__':
    option_1()
