import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import random
import keras
import random
import Helper_functions.helper_functions as help_fun

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

class sentiment_classifier:
    def __init__(self, file_name, seq_length, size_embedding, embedder):
        self.file_name = file_name
        self.seq_length = seq_length
        self.size_embedding = size_embedding
        self.embedder = embedder
        self.index_word = []
        self.history = []
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []

        try:
            self.model = self.load_model(file_name, seq_length, size_embedding)
        except:
            self.model = self.build_model(seq_length, size_embedding)

    # Build model to the specs
    def build_model(self, sequence_length, size_embedding):
        # what we want is for the output of the embedding to b
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(sequence_length, size_embedding)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(size_embedding)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.summary()
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.model

    # Train model with training data
    def train_model(self, num_epochs, batch_size):
        my_training_batch_generator = My_Custom_Generator(self.train_x, self.train_y, batch_size, self.embedder,
                                                          self.index_word,
                                                          self.size_embedding)

        my_validation_batch_generator = My_Custom_Generator(self.test_x, self.test_y, batch_size, self.embedder,
                                                            self.index_word,
                                                            self.size_embedding)

        num_train_samples = len(self.train_x)
        num_valid_samples = len(self.test_x)

        self.history = self.model.fit_generator(generator=my_training_batch_generator,
                                                steps_per_epoch=int(num_train_samples // batch_size),
                                                epochs=num_epochs,
                                                verbose=True,
                                                validation_data=my_validation_batch_generator,
                                                validation_steps=int(num_valid_samples // batch_size))

    def evaluate_model(self):
        print(self.model.evaluate(self.test_x, self.test_y))

    # Save model weights to file name
    def save_model(self, file_name):
        self.model.save_weights(file_name)

    # Load model from saved weights
    def load_model(self, file_name, sequence_length, size_embedding):
        model = self.build_model(sequence_length, size_embedding)
        model.load_weights(file_name)
        return model


    # Plot graphs to show how loss/ accuracy changed over time
    def plot_graphs(self, string):
        plt.plot(self.history.history[string])
        # plt.plot(history.history['val_'+string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        # plt.legend([string, 'val_'+string])
        plt.show()

    # needs to take in a Pandas DataFrame
    def pre_process_data(self, dataset):
        self.train_x, self.train_y, self.test_x, self.test_y = help_fun.pre_processing(dataset)

    def convert_data_pad_sequences(self):
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(self.train_x)
        self.index_word = tokenizer.index_word
        train_sequences = tokenizer.texts_to_sequences(self.train_x)
        train_padded_sentences = pad_sequences(train_sequences,
                                               padding='post', maxlen=self.seq_length)

        test_sequences = tokenizer.texts_to_sequences(self.test_x)
        test_padded_sentences = pad_sequences(test_sequences,
                                              padding='post', maxlen=self.seq_length)

        self.train_x = train_padded_sentences
        self.test_x = test_padded_sentences

    # needs to get passed in a sentence that is already vectorized
    def get_sentiment_with_index(self, sentences):
        print('Done Vectorizing, getting predictions')
        num_predictions = np.shape(sentences)[0]
        predictions = np.zeros((num_predictions, 1))
        idx_preds = np.arange(num_predictions)

        for idx, sentence in enumerate(sentences):
            prediction = self.model.predict(np.expand_dims(sentence, axis=0))
            predictions[idx] = prediction[0][0]

        print('Finished getting sentiments')
        return idx_preds, predictions

    def prepare_test_data(self):
        new_test_x = np.zeros((len(self.test_x), len(self.test_x[0]), self.size_embedding))
        for i in range(0, len(self.test_x)):
            num_words = len(self.test_x[i])
            for j in range(0, num_words):
                word = self.test_x[i][j]
                try:
                    word = self.index_word[word]
                    new_test_x[i][j] = self.embedder[word]
                except:
                    new_test_x[i][j] = self.embedder.wv[random.choice(self.embedder.wv.index2entity)]
        self.test_x = new_test_x

    def get_new_predictions(self, sentences):
        new_sentences = help_fun.prepare_new_predictions(sentences)
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(new_sentences)
        index_word = tokenizer.index_word
        new_sequences = tokenizer.texts_to_sequences(new_sentences)
        new_padded_sentences = pad_sequences(new_sequences,
                                             padding='post', maxlen=self.seq_length)

        print('Vectorizing New Sentences')
        new_sentences = np.zeros((len(new_padded_sentences), len(new_padded_sentences[0]), self.size_embedding))
        for i in range(0, len(new_padded_sentences)):
            num_words = len(new_padded_sentences[i])
            for j in range(0, num_words):
                word = new_padded_sentences[i][j]
                try:
                    word = index_word[word]
                    new_sentences[i][j] = self.embedder[word]
                except:
                    new_sentences[i][j] = self.embedder.wv[random.choice(self.embedder.wv.index2entity)]

        return self.get_sentiment_with_index(new_sentences)


class My_Custom_Generator(keras.utils.Sequence):
    def __init__(self, train_x, train_y, batch_size, embedder, word_index, embedding_size):
        self.train_x = train_x
        self.train_y = train_y
        self.batch_size = batch_size
        self.embedder = embedder
        self.word_index = word_index
        self.embedding_size = embedding_size

    def __len__(self):
        return (np.ceil(len(self.train_x) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        word_vectors = self.embedder.wv
        batch_x = self.train_x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.train_y[idx * self.batch_size: (idx + 1) * self.batch_size]

        new_batch_x = np.zeros((len(batch_x), len(batch_x[0]), self.embedding_size))

        for i in range(0, len(batch_x)):
            num_words = len(batch_x[i])
            for j in range(0, num_words):
                word = batch_x[i][j]
                try:
                    word = self.word_index[word]
                    new_batch_x[i][j] = self.embedder[word]

                except:
                    new_batch_x[i][j] = self.embedder.wv[random.choice(self.embedder.wv.index2entity)]

        batch_x = new_batch_x

        return batch_x, batch_y
