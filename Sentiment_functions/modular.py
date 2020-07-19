# INSTALL THESE

# To upgrade pip:
# pip install --upgrade pip
# NEED THESE: 
# pip install tensorflow
# pip install tensorflow-datasets

# Info on Tensorflow datasets: https://stackoverflow.com/questions/56820723/what-is-tensorflow-python-data-ops-dataset-ops-optionsdataset

#from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import KeyedVectors



## Define Constants here:
BUFFER_SIZE = 10000
BATCH_SIZE = 64
storage_location = Path('C:/Users/hsuen/Desktop/connected_journaling/connected_journaling/data/IMDB_Dataset.csv')
pre_trained_location = Path('C:/Users/hsuen/Desktop/bigData/GoogleNews-vectors-negative300.bin')


# Build model to the specs
def build_model(embedding_dim, word_index, sequence_length, pre_trained_location):
    # tokenizer = info.features['text'].encoder

    # use the pre-trained word-embeddings

    model = KeyedVectors.load_word2vec_format(pre_trained_location,
                                              binary=True)
    model = KeyedVectors.load_word2vec_format('C:\\Users\\hsuen\\Desktop\\bigData\\GoogleNews-vectors-negative300.bin',
                                              binary=True)

# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    #^ follow this example for how to use a pre-trained word embedding model
    model = tf.keras.Sequential([

        tf.keras.layers.Embedding(len(word_index), 64, input_size=sequence_length, weights=[embedding_matrix], trainable=False)

        #tf.keras.layers.Embedding(len(word_index), 64, trainable=False),


        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Train model with training data
def train_model(model, train_data, num_epochs):
    history = model.fit(train_data, epochs=num_epochs)
    return history


# Test model with test input: can be Numpy array, Tensorflow dataset
def test_model(model, test_data):
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    result = model.evaluate(test_data)
    print("test loss, test acc:", result)


# Save model weights to file name
def save_model(model, file_name):
    model.save_weights(file_name)


# Load model from saved weights
def load_model(file_name):
    model = create_model()
    model.load_weights(file_name)


# Plot graphs to show how loss/ accuracy changed over time
def plot_graphs(history, string):
    plt.plot(history.history[string])
    #plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    #plt.legend([string, 'val_'+string])
    plt.show()


# https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
# ^ follow this example to get the proper data

# gets passed in a .CSV file
def get_data(storage_location):
    #dataset, info = tfds.load('imdb_reviews/bytes', with_info=True, as_supervised=True)
    #train_dataset, test_dataset = dataset['train'], dataset['test']

    #train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    #train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
    #test_dataset = test_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset))
    #return train_dataset, test_dataset

    ############ Hermes Experimentation ##########
    dataset = pd.read_csv(storage_location)
    sentences = list(dataset.iloc[:,0])
    labels = list(dataset.iloc[:, 1])
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index # dictionary with the keys being the text items
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sentences = pad_sequences(sequences, padding='post') # numeric array of sequences that map on to actual word indices

    return padded_sentences, labels, word_index





train_dataset, test_dataset = get_data(storage_location)


# EXAMPLE OF CALLING ALL THE FUNCTIONS:
train_dataset,test_dataset = get_data()
our_model = build_model()
history = train_model(our_model, train_dataset)

#test_model(model, test_dataset)

#save_model(model, "/model_weights_trained")
#load_model("/model_weights_trained")

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')