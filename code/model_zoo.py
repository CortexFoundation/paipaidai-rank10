import numpy as np
import pandas as pd
from keras.layers import *
from keras.activations import softmax
from keras.models import Model,Sequential
from keras.optimizers import Nadam, Adam
from keras.regularizers import l2
import keras.backend as K

def cnn_lstm(word_embedding_matrix,num_shape,nb_words):
    MAX_SEQUENCE_LENGTH = 20
    MAX_NB_WORDS = 50000
    EMBEDDING_DIM = 300
    DROPOUT = 0.1
    filter_length = 5
    nb_filter = 64
    pool_length = 4
    OPTIMIZER = 'adam'

    question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

    e1 = Embedding(nb_words,
                     EMBEDDING_DIM,
                     weights=[word_embedding_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)(question1)

    q1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(e1)
    q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q1)

    e2 = Embedding(nb_words,
                     EMBEDDING_DIM,
                     weights=[word_embedding_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)(question2)
    q2 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(e2)
    q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q2)


    cnn1=Conv1D(filters=nb_filter,
                kernel_size=filter_length,
                padding='valid',
                activation='relu',
                strides=1)(e1)
    cnn1=Dropout(0.2)(cnn1)
    cnn1=Conv1D(filters=nb_filter,
                kernel_size=filter_length,
                padding='valid',
                activation='relu',
                strides=1)(cnn1)
    cnn1=GlobalMaxPooling1D()(cnn1)
    cnn1=Dropout(0.2)(cnn1)
    cnn1=Dense(300)(cnn1)
    cnn1=Dropout(0.2)(cnn1)
    cnn1=BatchNormalization()(cnn1)

    cnn2=Conv1D(filters=nb_filter,
                kernel_size=filter_length,
                padding='valid',
                activation='relu',
                strides=1)(e1)
    cnn2=Dropout(0.2)(cnn2)
    cnn2=Conv1D(filters=nb_filter,
                kernel_size=filter_length,
                padding='valid',
                activation='relu',
                strides=1)(cnn2)
    cnn2=GlobalMaxPooling1D()(cnn2)
    cnn2=Dropout(0.2)(cnn2)
    cnn2=Dense(300)(cnn2)
    cnn2=Dropout(0.2)(cnn2)
    cnn2=BatchNormalization()(cnn2)

    lstm1=LSTM(300, dropout=0.2, recurrent_dropout=0.2)(e1)
    lstm2=LSTM(300, dropout=0.2, recurrent_dropout=0.2)(e2)

    leaks_input = Input(shape=(num_shape,))
    leaks_dense = Dense(150, activation='relu')(leaks_input)

    merged = concatenate([q1,q2,cnn1,cnn2,lstm1,lstm2,leaks_dense])
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1,question2,leaks_input], outputs=is_duplicate)
    model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

    return model


def mlp(tfidf_shape,num_shape):
    DROPOUT=0.2
    tfidf_input = Input(shape=(tfidf_shape,))
    tfidf_dense = Dropout(DROPOUT)(tfidf_input)
    tfidf_dense = Dense(20, activation='relu')(tfidf_dense)

    leaks_input = Input(shape=(num_shape,))
    leaks_dense = Dense(20, activation='relu')(leaks_input)

    merged = concatenate([tfidf_dense,leaks_dense])
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[tfidf_input,leaks_input], outputs=is_duplicate)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model
