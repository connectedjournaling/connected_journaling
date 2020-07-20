### COPY TO GOOGLE COLLAB FOR LARGER DATASETS OF TRAINING
from pathlib import Path
import numpy as np
import sentiment_classifier as sc


## CONSTANTS
embedded_data_location = Path('C:/Users/hsuen/Desktop/connected_journaling/connected_journaling/data/imdb_1.npy')
label_data_location = Path('C:/Users/hsuen/Desktop/connected_journaling/connected_journaling/data/labels_imdb_1.npy')
output_model_location = 'C:/Users/hsuen/Desktop/connected_journaling/connected_journaling/Sentiment_functions/models/bad_model.h5'

embedding_size_constant = 300
epoch_constant = 1


print('loading data...')
train_data = np.load(embedded_data_location)
labels = np.load(label_data_location)

print('building model...')
model = sc.build_model(sequence_length=np.shape(train_data)[1], size_embedding=embedding_size_constant)
print('TRAINING MODEL...')
history, model = sc.train_model(model, train_data[:10], labels[:10], num_epochs=epoch_constant)
print('Plot training graphs...')
# plot_graphs(history, 'accuracy')
# plot_graphs(history, 'loss')

# this is an example prediction, it needs to have the minimum number of dimensinos be 3
result = model.predict(np.array(train_data[1][:][:], ndmin=3))

sc.save_model(model, output_model_location)


