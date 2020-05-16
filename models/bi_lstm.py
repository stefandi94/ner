import os

import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, Bidirectional
from keras.optimizers import Adam

from settings import VOCAB_SIZE


class BiLSTM:
    def __init__(self, sequence_length, n_neurons, n_classes):
        self.sequence_length = sequence_length
        self.n_neurons = n_neurons
        self.n_classes = n_classes
        self.model = None
        self.build_model()

    def build_model(self):
        input = Input(shape=(self.sequence_length,))
        layer = Embedding(input_dim=VOCAB_SIZE, output_dim=100, input_length=self.sequence_length, mask_zero=True)(input)
        layer = Bidirectional(LSTM(units=self.n_neurons, recurrent_dropout=0.2, dropout=0.2, return_sequences=True))(layer)
        output = TimeDistributed(Dense(units=self.n_classes, activation='softmax'))(layer)

        model = Model(input, output)
        model.compile(optimizer=Adam(learning_rate=0.01),
                      metrics=['acc'],
                      loss='categorical_crossentropy')
        self.model: Model = model

    def fit(self, X_train, y_train, X_valid, y_valid, epochs, batch_size, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        weights_name = "{epoch}-{loss:.3f}-{acc:.3f}-{val_loss:.3f}-{val_acc:.3f}.hdf5"
        checkpoint = ModelCheckpoint(os.path.join(save_dir, weights_name),
                                     monitor='val_acc',
                                     verbose=1,
                                     save_weights_only=False,
                                     save_best_only=True,
                                     mode='max')
        self.model.fit(X_train, y_train,
                       validation_data=(X_valid, y_valid),
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[checkpoint])

    def predict(self, X):
        X = np.array(X)
        if len(np.array(X).shape) == 1:
            X = np.reshape(X, newshape=(1, X.shape[0]))

        probabilities = self.model.predict(X)
        classes = probabilities.argmax(axis=2)
        return classes

    def load(self, path):
        self.model = self.model.load(path)
