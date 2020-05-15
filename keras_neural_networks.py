from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from keras.layers import Dense, Dropout, Activation, LSTM, Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import multiprocessing
import numpy as np
import pandas as pd
from sklearn import utils
from tqdm import tqdm

tqdm.pandas(desc="progress-bar")


class KerasNeuralNetwork:
    def __init__(self, X_train, X_test, vocabulary_size=100000):
        self.X_train = X_train
        self.X_test = X_test
        self.vocabulary_size = vocabulary_size
        self.max_sentence_length = None

    def build_embedding_matrix(self):
        print('Building Word2Vec embeeding matrix')

        def _make_tags_for_sentences(dataset, label):
            all_tagged_documents = []
            for index, sentence in zip(dataset.index, dataset):
                all_tagged_documents.append(TaggedDocument(sentence.split(), [label + f'_{index}']))
            return all_tagged_documents

        all_x_w2v = _make_tags_for_sentences(pd.concat([self.X_train, self.X_test]), 'all')

        cores = multiprocessing.cpu_count()
        model_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065,
                              min_alpha=0.065)
        model_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])

        for epoch in range(30):
            model_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
            model_cbow.alpha -= 0.002
            model_cbow.min_alpha = model_cbow.alpha

        model_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065,
                            min_alpha=0.065)
        model_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])

        for epoch in range(30):
            model_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
            model_sg.alpha -= 0.002
            model_sg.min_alpha = model_sg.alpha

        # model_cbow.save('w2v_model_ug_cbow.word2vec')
        # model_sg.save('w2v_model_ug_sg.word2vec')

        embeddings_index = {}
        for w in model_cbow.wv.vocab.keys():
            embeddings_index[w] = np.append(model_cbow.wv[w], model_sg.wv[w])
        print(f'Found {len(embeddings_index)} word vectors')

        tokenizer = Tokenizer(num_words=self.vocabulary_size)
        tokenizer.fit_on_texts(self.X_train)
        embedding_matrix = np.zeros((self.vocabulary_size, 200))
        for word, i in tokenizer.word_index.items():
            if i >= self.vocabulary_size:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return [embedding_matrix]

    def build_CNN(self, shape, number_of_classes, word2vec_weights=None, trainable=None, dropout=0.5):
        dimensions_for_vector_representation_of_word = 200
        node = 512 * 2
        hidden_layers = 1

        model_cnn = Sequential()
        if word2vec_weights:
            # The embedding layer encodes the input sequence into a sequence of dense vectors of this dimension
            e = Embedding(self.vocabulary_size, dimensions_for_vector_representation_of_word, weights=word2vec_weights,
                          input_length=self.max_sentence_length, trainable=trainable)
            model_cnn.add(e)
        else:
            e = Embedding(self.vocabulary_size, dimensions_for_vector_representation_of_word,
                          input_length=self.max_sentence_length)
            model_cnn.add(e)
        model_cnn.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
        model_cnn.add(GlobalMaxPooling1D())
        model_cnn.add(Dense(node, input_dim=shape, activation='relu'))
        model_cnn.add(Dropout(dropout))
        for i in range(hidden_layers):
            model_cnn.add(Dense(node, activation='relu'))
            model_cnn.add(Dropout(dropout))
        model_cnn.add(Dense(number_of_classes, activation='softmax'))
        model_cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_cnn.summary()
        return model_cnn

    def build_RCNN(self, shape, number_of_classes, word2vec_weights=None, trainable=None, dropout=0.5):
        dimensions_for_vector_representation_of_word = 200
        node = 512 * 2
        gru_node = 256
        filters = 256
        conv_layers = 4
        lstm_layers = 3
        pool_size = 2

        model_rcnn = Sequential()
        if word2vec_weights:
            e = Embedding(self.vocabulary_size, dimensions_for_vector_representation_of_word, weights=word2vec_weights,
                          input_length=self.max_sentence_length, trainable=trainable)
            model_rcnn.add(e)
        else:
            e = Embedding(self.vocabulary_size, dimensions_for_vector_representation_of_word,
                          input_length=self.max_sentence_length)
            model_rcnn.add(e)

        model_rcnn.add(Dropout(dropout))
        for i in range(conv_layers):
            model_rcnn.add(Conv1D(filters, kernel_size=2, activation='relu'))
            model_rcnn.add(MaxPooling1D(pool_size=pool_size))

        model_rcnn.add(Dense(node, input_dim=shape, activation='relu'))
        model_rcnn.add(Dropout(dropout))

        for i in range(lstm_layers):
            model_rcnn.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=dropout))

        model_rcnn.add(LSTM(gru_node, recurrent_dropout=dropout))
        model_rcnn.add(Dense(node, activation='relu'))
        model_rcnn.add(Dropout(dropout))
        model_rcnn.add(Dense(number_of_classes))
        model_rcnn.add(Activation('softmax'))
        model_rcnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_rcnn.summary()
        return model_rcnn

    def build_LSTM(self, number_of_classes, dropout=0.5):
        embed_dim = 200
        # The LSTM transforms the vector sequence into a single vector of size lstm_hidden_nodes, containing information
        # about the entire sequence
        lstm_hidden_nodes = 200
        hidden_layers = 1
        hidden_nodes = 512 * 2

        model_lstm = Sequential()
        model_lstm.add(Embedding(self.vocabulary_size, embed_dim, input_length=self.max_sentence_length))
        # return_sequences=True argument ensures that the LSTM cell returns all of the outputs from the unrolled LSTM
        # cell through time
        model_lstm.add(LSTM(lstm_hidden_nodes, dropout=0.5, recurrent_dropout=0.5))
        for i in range(hidden_layers):
            model_lstm.add(Dense(hidden_nodes, activation='relu'))
            model_lstm.add(Dropout(dropout))
        model_lstm.add(Dense(number_of_classes, activation='softmax'))
        model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_lstm.summary()
        return model_lstm

    def build_DNN(self, shape, number_of_classes, dropout=0.5):
        node = 512 * 2
        hidden_layers = 3

        model_dnn = Sequential()
        model_dnn.add(Dense(node, input_dim=shape, activation='relu'))
        # Dropout is simulating as if we train many different networks and averaging them by randomly omitting hidden
        # nodes with a certain probability throughout the training process.
        model_dnn.add(Dropout(dropout))
        for i in range(hidden_layers):
            model_dnn.add(Dense(node, activation='relu'))
            model_dnn.add(Dropout(dropout))
        model_dnn.add(Dense(number_of_classes, activation='softmax'))
        model_dnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_dnn.summary()
        return model_dnn

    def tokenize(self):
        tokenizer = Tokenizer(num_words=self.vocabulary_size)
        tokenizer.fit_on_texts(self.X_train)
        sequences_train = tokenizer.texts_to_sequences(self.X_train)
        sequences_test = tokenizer.texts_to_sequences(self.X_test)
        self.max_sentence_length = max(map(len, sequences_train + sequences_test)) + 5
        x_train_seq = pad_sequences(sequences_train, maxlen=self.max_sentence_length)
        x_test_seq = pad_sequences(sequences_test, maxlen=self.max_sentence_length)
        return x_train_seq, x_test_seq
