import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import KeyedVectors
import numpy as np

## Define Constants here:
BATCH_SIZE = 64
storage_location_path = Path(
    'C:/Users/hsuen/Desktop/connected_journaling/connected_journaling/data/IMDB_Dataset.csv')  # train data



# Build model to the specs
def build_model(sequence_length, size_embedding):
    # what we want is for the output of the embedding to b
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(sequence_length, size_embedding)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(size_embedding)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Train model with training data
def train_model(model, x_data, y_data, num_epochs):
    history = model.fit(x_data, y_data, epochs=num_epochs, validation_split=0.2, verbose=True)
    return history, model


# Save model weights to file name
def save_model(model, file_name):
    model.save_weights(file_name)


# Load model from saved weights
def load_model(file_name, sequence_length, size_embedding):
    model = build_model(sequence_length, size_embedding)
    model.load_weights(file_name)
    return model



# Plot graphs to show how loss/ accuracy changed over time
def plot_graphs(history, string):
    plt.plot(history.history[string])
    # plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    # plt.legend([string, 'val_'+string])
    plt.show()


# gets passed in a .CSV file for training
# gets passed in list of sentences for test
def convert_data_train(storage_location, pre_trained_location, size_embedding):
    print('Pre-processing data...')
    dataset = pd.read_csv(storage_location)

    dataset = dataset[10:][:] ## UNCOMMENT THIS LINE FOR REDUCING THE DATA SET

    sentences = list(dataset.iloc[:, 0])
    labels = list(dataset.iloc[:, 1])
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)

    index_word = tokenizer.index_word  # dictionary with the indexes as the keys
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sentences = pad_sequences(sequences,
                                     padding='post')  # numeric array of sequences that map on to actual word indices

    print('Loading Pre-Trained...')
    word_vectors = KeyedVectors.load_word2vec_format(pre_trained_location,
                                                     binary=True)
    embedded_data_tensor = np.zeros((len(padded_sentences), len(padded_sentences[0]), size_embedding))
    for observation, sentence in enumerate(padded_sentences):
        for time_step, word in enumerate(sentence):
            try:
                word_vector = word_vectors[index_word[word]]
                embedded_data_tensor[observation][time_step][:] = word_vector
            except:
                # word is either zero or not included in the pre-trained embedding dimension
                embedded_data_tensor[observation][time_step][:] = np.zeros((1, size_embedding))

        print('Index {0}'.format(observation))
    return embedded_data_tensor, labels


def convert_data_test(sentences, sequence_length, pre_trained_location, size_embedding):
    ## eliminate half of the dataset for memory purposes:
    sentences = sentences[:100]

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)

    index_word = tokenizer.index_word  # dictionary with the indexes as the keys
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sentences = pad_sequences(sequences,
                                     padding='post', maxlen=sequence_length)

    print('Loading Pre-Trained...')
    word_vectors = KeyedVectors.load_word2vec_format(pre_trained_location,
                                                     binary=True)

    embedded_data_tensor = np.zeros((len(padded_sentences), len(padded_sentences[0]), size_embedding))
    for observation, sentence in enumerate(padded_sentences):
        for time_step, word in enumerate(sentence):
            try:
                word_vector = word_vectors[index_word[word]]
                embedded_data_tensor[observation][time_step][:] = word_vector
            except:
                # word is either zero or not included in the pre-trained embedding dimension
                embedded_data_tensor[observation][time_step][:] = np.zeros((1, size_embedding))

        print('Index {0}'.format(observation))
    return embedded_data_tensor


# needs to get passed in a sentence that is already vectorized
def get_sentiment_with_index(sentences, model):
    num_predictions = np.shape(sentences)[0]
    predictions = np.zeros(num_predictions, 1)
    idx_preds = np.arange(num_predictions)

    for idx, sentence in enumerate(sentences):
        prediction = model.predict(np.array(sentence[idx][:][:], ndmin=3))
        predictions[idx] = prediction
    
    return idx_preds, predictions

