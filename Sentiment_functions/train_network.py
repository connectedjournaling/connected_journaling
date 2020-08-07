### COPY TO GOOGLE COLLAB FOR LARGER DATASETS OF TRAINING
from pathlib import Path
import Sentiment_functions.sentiment_classifier as sc
import pandas as pd
from gensim.models import KeyedVectors


## CONSTANTS ##
storage_location = Path("C:/Users/hsuen/Desktop/connected_journaling/connected_journaling/data/IMDB_Dataset.csv")
pre_trained_embeddings_path = Path('C:/Users/hsuen/Desktop/bigData/GoogleNews-vectors-negative300.bin')
sentiment_network_path = 'C:/Users/hsuen/Desktop/connected_journaling/connected_journaling/Sentiment_functions/models/good_model.h5'


print('loading in word vectors')
word_vectors = KeyedVectors.load_word2vec_format(pre_trained_embeddings_path,
                                                         binary=True)

embedding_size_constant = 300
sequence_length = 150
num_epochs = 60
batch_size = 32

## Train Model ##
model = sc.sentiment_classifier(sentiment_network_path, sequence_length, embedding_size_constant, word_vectors)

dataset = pd.read_csv(storage_location)

model.pre_process_data(dataset)
model.convert_data_pad_sequences()
model.train_model(num_epochs, batch_size)
model.prepare_test_data()
model.evaluate_model()

model.save_model(sentiment_network_path)


######
#model.load_model(sentiment_network_path, sequence_length, embedding_size_constant)
#idx, preds = model.get_new_predictions(['FUCK you', 'i love this'])