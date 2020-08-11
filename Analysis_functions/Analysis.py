import cluster_important_words as cluster
import sentiment_classifier as sentiment
import helper_functions as help_fun
import Syntax_functions.class_defs.NER as NER
import Syntax_functions.class_defs.DPR as DPR
from gensim.models import KeyedVectors
import pandas as pd
import spacy


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
        self.ner_model = NER.NER("en_core_web_md")
        self.dpr_model = DPR.DPR("en_core_web_md")

        ## initialize data variables that are handled by the class ##
        self.sentences = []
        self.num_sentences = 0

        self.imp_words = []
        self.num_groups = 0

        self.entities = {}
        self.noun_objects = []
        self.noun_subjects = []

        self.sentiments = []
        self.sentiment_model = []
        self.vectorizer = []

        ## then run an initializing function to get things rolling ##
        self._get_things_ready()

    def _get_things_ready(self):
        print('READING IN CSV FILE')
        self.data = pd.read_csv(self.data_path)
        self.data = self.data['text']

        print('CONVERTING DATA INTO SENTENCES')
        self.sentences = help_fun.split_into_sentence(self.data_path)
        self.sentences_for_sentiment = []
        self.num_sentences = len(self.sentences)

        print('LOADING THE VECTORIZER')
        # load in a pre-trained word-embedding from google's word2vec
        self.vectorizer = KeyedVectors.load_word2vec_format(self.embedding_path,
                                                            binary=True)

        print('LOADING TRAINED SENTIMENT CLASSIFIER')
        self.sentiment_model = sentiment.sentiment_classifier(self.sentiment_model_path, self.sequence_length,
                                                              self.embedding_size, self.vectorizer)

        print('########DONE INITIALIZING#####')

    def get_imp_words(self):
        imp_word_list, num_groups = cluster.get_important_words(self.data, self.vectorizer)
        self.imp_words = imp_word_list
        self.num_groups = num_groups

    def get_all_entities(self):
        for idx, sentence in enumerate(self.sentences):
            entities = self.ner_model.evaluate(sentence)
            for entity in entities:
                if entity[0] not in self.entities:
                    self.entities[entity[0]] = []
                    self.entities[entity[0]].append(idx)
                elif entity[0] in self.entities:
                    if idx not in self.entities[entity[0]]:
                        self.entities[entity[0]].append(idx)

    def get_all_sentiments(self):
        print('Starting Sentiment Analysis')
        idx_preds, predictions = self.sentiment_model.get_new_predictions(self.sentences)
        self.sentiments = predictions

    def get_all_objects(self):
        for idx, sentence in enumerate(self.sentences):
            sentence_parts = self.dpr_model.get_all(sentence)
            noun_objs = sentence_parts['Objects']
            self.noun_objects.append(noun_objs)

    def get_all_subjects(self):
        for idx, sentence in enumerate(self.sentences):
            sentence_parts = self.dpr_model.get_all(sentence)
            noun_objs = sentence_parts['Subjects']
            self.noun_subjects.append(noun_objs)

    def get_top_sentences(self, threshold):
        if len(self.sentiments) == 0:
            print('Have not extracted Sentiments')
            return -1

        top_sentences = self.sentiments > threshold
        return 0
