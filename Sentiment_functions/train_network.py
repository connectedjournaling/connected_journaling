### COPY TO GOOGLE COLLAB FOR LARGER DATASETS OF TRAINING
from pathlib import Path
# import sentiment_classifier as sc

import Sentiment_functions.sentiment_classifier as sc
import pandas as pd
from gensim.models import KeyedVectors


## CONSTANTS ##
#storage_location = Path("C:/Users/hsuen/Desktop/connected_journaling/connected_journaling/data/IMDB_Dataset.csv")
storage_location = Path("/Users/petergramaglia/Documents/GitHub/new_connected/connected_journaling/data/IMDB_Dataset_Tiny.csv")
pre_trained_embeddings_path = Path("/Users/petergramaglia/Documents/GitHub/new_connected/connected_journaling/data/GoogleNews-vectors-negative300.bin")
#pre_trained_embeddings_path = Path('C:/Users/hsuen/Desktop/bigData/GoogleNews-vectors-negative300.bin')
sentiment_network_path = 'C:/Users/hsuen/Desktop/connected_journaling/connected_journaling/Sentiment_functions/models/good_model.h5'


print('loading in word vectors')
word_vectors = KeyedVectors.load_word2vec_format(pre_trained_embeddings_path,
                                                         binary=True)

embedding_size_constant = 300
#embedding_size_constant = 128 # Experimenting
sequence_length = 150
num_epochs = 6
batch_size = 32

## Train Model ##
sentiment_network_path = "nada" # For when you don't want to load pre-trained models
model = sc.sentiment_classifier(sentiment_network_path, sequence_length, embedding_size_constant, word_vectors)

print("Reading csv")
dataset = pd.read_csv(storage_location)

print("Preprocessing")
model.pre_process_data(dataset)

print("Converting")
model.convert_data_pad_sequences()

print("Training")
model.train_model(num_epochs, batch_size)

print("Prepare test")
model.prepare_test_data()

print("Evaluating")
model.evaluate_model()

#model.save_model(sentiment_network_path)


######
#model.load_model(sentiment_network_path, sequence_length, embedding_size_constant)
#idx, preds = model.get_new_predictions(['FUCK you', 'i love this'])