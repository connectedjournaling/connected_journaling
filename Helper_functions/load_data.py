class My_Custom_Generator(keras.utils.Sequence):

    # csv_file_train_data has two columns, one is "text", other is "label"
    # load the word embedding .bin file in to
    def __init__(self, csv_file_train_data, bin_file_word_embeddings, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.model = # load pre-trained embedding model here

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    # takes in "idx" representing mini-batch number, and returns a tuple:
    # 3-D matrix minibatchsize x words in sentence x 300
    # labels: mini batch size x 1 vector
    # convert text to numpy arrays
    def __getitem__(self, idx):
        self.model(word)
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        return np.array([
            resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
            for file_name in batch_x]) / 255.0, np.array(batch_y)