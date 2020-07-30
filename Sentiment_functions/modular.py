# INSTALL THESE

# To upgrade pip:
# pip install --upgrade pip
# NEED THESE: 
# pip install tensorflow
# pip install tensorflow-datasets

# Info on Tensorflow datasets: https://stackoverflow.com/questions/56820723/what-is-tensorflow-python-data-ops-dataset-ops-optionsdataset

from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow_datasets as tfds
import tensorflow as tf
#print(tf.__version__)s
import matplotlib.pyplot as plt

BUFFER_SIZE = 10000
BATCH_SIZE = 64

# Build model to the specs
def build_model: 
    tokenizer = info.features['text'].encoder

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train model with training data
def train_model(model, train_data):
    NUM_EPOCHS = 50
    history = model.fit(train_data, epochs=NUM_EPOCHS)
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

def get_data():
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
    test_dataset = test_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset))
    return train_dataset, test_dataset




# EXAMPLE OF CALLING ALL THE FUNCTIONS:
train_dataset,test_dataset = get_data()
our_model = build_model()
history = train_model(our_model, train_dataset)

test_model(model, test_dataset)

#save_model(model, "/model_weights_trained")
#load_model("/model_weights_trained")

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')