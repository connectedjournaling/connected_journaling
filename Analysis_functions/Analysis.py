import cluster_important_words as cluster
import sentiment_classifier as sentiment
import helper_functions as help_fun
from gensim.models import KeyedVectors
import pandas as pd


class WeekAnalysis:
    def __init__(self, data_path, embedding_path, embedding_size, sentiment_model, sequence_length):
        print('########STARTING INITIALIZATION########')
        ## define all of the constants, variables, functions, and classes within WeekAnalysis ##
        self.data_path = data_path
        self.embedding_path = embedding_path
        self.sentiment_model_path = sentiment_model
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.cluster_funcs = cluster
        self.sentiment_funcs = sentiment

        ## initialize data variables that are handled by the class ##
        self.sentences = []
        self.num_sentences = 0
        self.imp_words = []
        self.num_groups = 0
        self.entities = []
        self.noun_objects = []
        self.sentiments = []
        self.sentiment_model = []
        self.vectorizer = []
        self.train_data = []

        ## then run an initializing function to get things rolling ##
        self._get_things_ready()

    def _get_things_ready(self):
        print('READING IN CSV FILE')
        self.train_data = pd.read_csv(self.data_path)
        self.train_data = self.train_data['text']

        print('CONVERTING DATA INTO SENTENCES')
        self.sentences = help_fun.split_into_sentence(self.data_path)

        print('LOADING TRAINED SENTIMENT CLASSIFIER')
        self.sentiment_model = sentiment.load_model(
            self.sentiment_model_path,
            self.sequence_length, self.embedding_size)

        print('LOADING THE VECTORIZER')
        # load in a pre-trained word-embedding from google's word2vec
        self.vectorizer = KeyedVectors.load_word2vec_format(self.embedding_path,
                                                            binary=True)

        print('########DONE INITIALIZING#####')

    def get_imp_words(self):
        imp_word_list, num_groups = cluster.get_important_words(self.train_data, self.vectorizer)
        self.imp_words = imp_word_list
        self.num_groups = num_groups

    def get_sentiments(self):
