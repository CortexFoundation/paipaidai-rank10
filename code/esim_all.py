import numpy as np
import pandas as pd
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras.regularizers import l2
import keras.backend as K

def create_pretrained_embedding(pretrained_weights, trainable=False, **kwargs):
    "Create embedding layer from a pretrained weights array"
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=False, **kwargs)
    return embedding


def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_ = Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def esim(pretrained_weights_char,
         pretrained_weights_word,
         num_shape,
         maxlen_char=40,
         maxlen_word=20,
         lstm_dim=300,
         dense_dim=300,
         dense_dropout=0.2):
    # Based on arXiv:1609.06038
    ############char##################
    q1 = Input(shape=(maxlen_char,))
    q2 = Input(shape=(maxlen_char,))

    # Embedding
    embedding = create_pretrained_embedding(pretrained_weights_char, mask_zero=False)
    bn = BatchNormalization(axis=2)
    q1_embed = bn(embedding(q1))
    q2_embed = bn(embedding(q2))

    # Encode
    encode = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compose
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

    compose = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)

    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    ############word##################
    q3 = Input(shape=(maxlen_word,))
    q4 = Input(shape=(maxlen_word,))

    # Embedding
    embedding = create_pretrained_embedding(pretrained_weights_word, mask_zero=False)
    bn = BatchNormalization(axis=2)
    q3_embed = bn(embedding(q3))
    q4_embed = bn(embedding(q4))

    # Encode
    encode = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q3_encoded = encode(q3_embed)
    q4_encoded = encode(q4_embed)

    # Attention
    q3_aligned, q4_aligned = soft_attention_alignment(q3_encoded, q4_encoded)

    # Compose
    q3_combined = Concatenate()([q3_encoded, q4_aligned, submult(q3_encoded, q4_aligned)])
    q4_combined = Concatenate()([q4_encoded, q3_aligned, submult(q4_encoded, q3_aligned)])

    compose = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q3_compare = compose(q3_combined)
    q4_compare = compose(q4_combined)

    # Aggregate
    q3_rep = apply_multiple(q3_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q4_rep = apply_multiple(q4_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])


    leaks_input = Input(shape=(num_shape,))
    leaks_dense = Dense(dense_dim//2, activation='relu')(leaks_input)

    dot_input=[dot([q1_rep,q3_rep],axes=1),
               dot([q2_rep,q4_rep],axes=1),
               dot([q1_rep,q2_rep],axes=1),
               dot([q3_rep,q4_rep],axes=1)]
    sub_input=[Lambda(lambda x: K.abs(x[0] - x[1]))([q1_rep,q3_rep]),
               Lambda(lambda x: K.abs(x[0] - x[1]))([q2_rep,q4_rep]),
               Lambda(lambda x: K.abs(x[0] - x[1]))([q1_rep,q2_rep]),
               Lambda(lambda x: K.abs(x[0] - x[1]))([q3_rep,q4_rep])]
    # Classifier
    merged = Concatenate()([q1_rep,q2_rep,q3_rep,q4_rep,leaks_dense]+dot_input+sub_input)

    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2,q3,q4,leaks_input], outputs=out_)
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['binary_crossentropy', 'accuracy'])
    return model
