print("Importing..............")
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import random
import string
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from gensim.models import KeyedVectors

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


## LOAD CONSTANTS HERE ##
table = str.maketrans('', '', string.punctuation)
data_path = '/Users/petergramaglia/Documents/GitHub/new_connected/connected_journaling/data/IMDB_Dataset_Small.csv'
output_model_location = '/Users/petergramaglia/Documents/GitHub/new_connected/connected_journaling/Sentiment_functions/models/bad_model.h5'

print("Loading embedder.......")
# Embedder
embedder = KeyedVectors.load_word2vec_format('/Users/petergramaglia/Documents/GitHub/new_connected/connected_journaling/data/GoogleNews-vectors-negative300.bin',binary=True)
word_vectors = embedder.wv

print("Reading dataset........")
dataset = pd.read_csv(data_path)   # 5k samples

print("Shuffling/ splitting...")
# Shuffle, train/test split
shuffled_dataset = shuffle(dataset)
train_dataset, test_dataset = train_test_split(
    shuffled_dataset, test_size=0.5, random_state=1)


print("Dividing...............")
# Divide up x and y
#print(train_dataset[0:10])
train_x = train_dataset.iloc[:,0]
train_y = train_dataset.iloc[:,1]
test_x = test_dataset.iloc[:,0]
test_y = test_dataset.iloc[:,1]
#print(train_x[0:10])


train_x = train_x.to_numpy()
train_y = train_y.to_numpy()
test_x = test_x.to_numpy()
test_y = test_y.to_numpy()


print("Pre-processing.........")
for i in range(0,len(train_x)):
    #print(train_x[i])
    train_x[i] = train_x[i].replace('<br />', '')
    test_x[i] = test_x[i].replace('<br />', '')

for i in range(0,len(train_x)):
    train_x[i] = train_x[i].split()
    train_x[i] = [word.translate(table) for word in train_x[i]]
    train_x[i] = [word.lower() for word in train_x[i]]
    train_x[i] = [word for word in train_x[i] if word not in stopwords.words('english')]

    test_x[i] = test_x[i].split()
    test_x[i] = [word.translate(table) for word in test_x[i]]
    test_x[i] = [word.lower() for word in test_x[i]]
    test_x[i] = [word for word in test_x[i] if word not in stopwords.words('english')]

#print(batch_x[0])
print("Mas - padding")
input_length = 150
for i in range(0,len(train_x)):
    diff = len(train_x[i]) - input_length
    #print("diff:",diff)
    if diff < 0:
        for j in range(0,np.abs(diff)):
            train_x[i].append("## ")
    elif diff > 0:
        train_x[i] = train_x[i][:input_length]

    diff = len(test_x[i]) - input_length
    #print("diff:",diff)
    if diff < 0:
        for j in range(0,np.abs(diff)):
            test_x[i].append("## ")
    elif diff > 0:
        test_x[i] = test_x[i][:input_length]


# batch_x_pad = pad_sequences(
#     sequences=batch_x,
#     maxlen=input_length,
#     padding='post')


print("Y stuff")
new_train_y = np.zeros(len(train_y))
new_test_y = np.zeros(len(test_y))

for i in range(0,len(train_y)):
    if train_y[i] == "positive":
        new_train_y[i] = 1
    if test_y[i] == "positive":
        new_test_y[i] = 1

# These next two lines don't do anything important – just managing
# variable names for easier reading.
train_y = new_train_y
test_y = new_test_y

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_x)

# index_word = tokenizer.index_word  # dictionary with the indexes as the keys
# sequences = tokenizer.texts_to_sequences(sentences)



class My_Custom_Generator(keras.utils.Sequence) :
    def __init__(self, train_x, train_y, batch_size) :
        self.train_x = train_x
        self.train_y = train_y
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.train_x) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_x = train_x[idx * batch_size : (idx+1) * batch_size]
        batch_y = train_y[idx * batch_size : (idx+1) * batch_size]


        new_batch_x = np.zeros((len(batch_x), len(batch_x[0]), 300))

        for i in range(0,len(batch_x)):
            #print("i:", i)
            num_words = len(batch_x[i])
            #print(num_words)
            #print(batch_x[i])
            for j in range(0,num_words):
                #print("j: ", j)
                word = batch_x[i][j]
                if word in word_vectors.vocab:
                    new_batch_x[i][j] = embedder[word]
                    #print(batch_x.shape)
                else:
                    new_batch_x[i][j] = embedder.wv[random.choice(embedder.wv.index2entity)]

        #batch_x = np.expand_dims(batch_x, -1)
        #batch_y = np.expand_dims(batch_y, -1)
        batch_x = new_batch_x

        return batch_x, batch_y


print("Making generators")
batch_size = 32

my_training_batch_generator = My_Custom_Generator(train_x, train_y, batch_size)
my_validation_batch_generator = My_Custom_Generator(test_x, test_y, batch_size)



# model = tf.keras.Sequential([
#     #tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

print("Building model")
input_shape = embedder["dog"].shape
print(input_shape)
model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, input_shape = input_shape, return_sequences=True)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
])

num_train_samples = len(train_x)
num_valid_samples = len(test_x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.summary()

model.fit_generator(generator=my_training_batch_generator,
                   steps_per_epoch = int(num_train_samples // batch_size),
                   epochs = 6,
                   verbose = 1,
                   validation_data = my_validation_batch_generator,
                   validation_steps = int(num_valid_samples // batch_size))



print("Doing test embeddings")
new_test_x = np.zeros((len(test_x), len(test_x[0]), 300))
for i in range(0,len(test_x)):
            num_words = len(test_x[i])
            #print(num_words)
            #print(batch_x[i])
            for j in range(0,num_words):
                #print("j: ", j)
                word = test_x[i][j]
                if word in word_vectors.vocab:
                    new_test_x[i][j] = embedder[word]
                    #print(batch_x.shape)
                else:
                    new_test_x[i][j] = embedder.wv[random.choice(embedder.wv.index2entity)]

test_x = new_test_x

print("Evaluate on test data")
result = model.evaluate(test_x, test_y)







