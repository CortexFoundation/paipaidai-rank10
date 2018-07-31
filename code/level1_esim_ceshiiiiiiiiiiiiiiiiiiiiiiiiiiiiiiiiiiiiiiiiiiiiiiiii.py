# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import time
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from gensim.models import word2vec
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Activation, Dropout, Embedding,BatchNormalization,Bidirectional,Conv1D,GlobalMaxPooling1D,Input,Lambda,TimeDistributed,Convolution1D
from keras.layers import LSTM,concatenate
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

t1=time.time()
###################################################################################################################################
#get embedding
emb_dic={}
with open("../input/word_embed.txt") as f:
    word_emb=f.readlines()
    word_emb=word_emb
    print(len(word_emb))
    for w in word_emb:
        w=w.replace("\n","")
        content=w.split(" ")
        emb_dic[content[0].lower()]=np.array(content[1:])

MAX_SEQUENCE_LENGTH = 20
MAX_NB_WORDS = 50000
EMBEDDING_DIM = len(content)-1
DROPOUT = 0.1
###################################################################################################################################
#get data
train = pd.read_csv('../input/train.csv')#[:10000]
test = pd.read_csv('../input/test.csv')#[:10000]
ques=pd.read_csv('../input/question.csv')
ques.columns=["q1","w1","c1"]
train=train.merge(ques,on="q1",how="left")
test=test.merge(ques,on="q1",how="left")
ques.columns=["q2","w2","c2"]
train=train.merge(ques,on="q2",how="left")
test=test.merge(ques,on="q2",how="left")

#############################################################################################################################
#MAGIC_FEATURE
train_df = pd.read_csv("../input/train.csv")#[:10000]
test_df = pd.read_csv("../input/test.csv")#[:10000]
test_df["label"]=-1
data = pd.concat([train_df[['q1', 'q2']], \
                  test_df[['q1', 'q2']]], axis=0).reset_index(drop='index')
q_dict = defaultdict(set)
for i in range(data.shape[0]):
    q_dict[data.q1[i]].add(data.q2[i])
    q_dict[data.q2[i]].add(data.q1[i])
def q1_freq(row):
    return (len(q_dict[row['q1']]))
def q2_freq(row):
    return (len(q_dict[row['q2']]))
def q1_q2_intersect(row):
    return (len(set(q_dict[row['q1']]).intersection(set(q_dict[row['q2']]))))
train_df['q1_q2_intersect'] = train_df.apply(q1_q2_intersect, axis=1, raw=True)
train_df['q1_freq'] = train_df.apply(q1_freq, axis=1, raw=True)
train_df['q2_freq'] = train_df.apply(q2_freq, axis=1, raw=True)
test_df['q1_q2_intersect'] = test_df.apply(q1_q2_intersect, axis=1, raw=True)
test_df['q1_freq'] = test_df.apply(q1_freq, axis=1, raw=True)
test_df['q2_freq'] = test_df.apply(q2_freq, axis=1, raw=True)

leaks = train_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]
test_leaks = test_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]

ss = StandardScaler()
ss.fit(np.vstack((leaks, test_leaks)))
leaks = ss.transform(leaks)
test_leaks = ss.transform(test_leaks)
#############################################################################################################################
#process data
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS,)
tokenizer.fit_on_texts(list(train["w1"])+list(test["w1"])+list(train["w2"])+list(test["w2"]))
column="w1"
sequences_all = tokenizer.texts_to_sequences(list(train[column]))
sequences_test = tokenizer.texts_to_sequences(list(test[column]))
X_train_1 = pad_sequences(sequences_all, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
X_test_1 = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
column="w2"
sequences_all = tokenizer.texts_to_sequences(list(train[column]))
sequences_test = tokenizer.texts_to_sequences(list(test[column]))
X_train_2 = pad_sequences(sequences_all, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
X_test_2 = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH,padding='post')

word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index))+1
print(nb_words)


ss=0
word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
print(len(word_index.items()))
for word, i in word_index.items():
    if word in emb_dic.keys():
        ss+=1
        word_embedding_matrix[i] = emb_dic[word]
    else:
        pass
print(ss)
print(word_embedding_matrix)
y=train["label"]
print(y.value_counts())
###################################################################################################################################
# 建立模型
from keras import *
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras.regularizers import l2
import keras.backend as K
from sklearn.cross_validation import StratifiedKFold,KFold
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

def build_model():
    emb_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[word_embedding_matrix],
                          input_length=MAX_SEQUENCE_LENGTH, trainable=True)

    # Define inputs
    seq1 = Input(shape=(20,))
    seq2 = Input(shape=(20,))

    # Run inputs through embedding
    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)

    lstm_layer = Bidirectional(LSTM(300, dropout=0.15, recurrent_dropout=0.15, return_sequences=True))
    lstm_layer2 = Bidirectional(LSTM(300, dropout=0.15, recurrent_dropout=0.15))
    # lstm_layer3 = Bidirectional(LSTM(300, dropout=0.15, recurrent_dropout=0.15))

    que_1 = lstm_layer(emb1)
    ans_1 = lstm_layer(emb2)
    que = lstm_layer2(que_1)
    ans = lstm_layer2(ans_1)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(que_1, ans_1)

    # Compose
    q1_combined = Concatenate()([que_1, q2_aligned, submult(que_1, q2_aligned)])
    q2_combined = Concatenate()([que_1, q1_aligned, submult(ans_1, q1_aligned)])
    q1_rep = apply_multiple(q1_combined, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_combined, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    mul = layers.multiply([que, ans])
    sub = layers.subtract([que, ans])
    diff = Lambda(lambda x: K.abs(x[0] - x[1]))([que, ans])
    add = layers.add([que, ans])
    #merge = concatenate([que, ans, mul, sub,diff,add])

    leaks_input = Input(shape=(3,))
    leaks_dense = Dense(150, activation='relu')(leaks_input)

    merge = concatenate([mul, sub, diff,q1_rep,q2_rep,leaks_dense])


    x = Dropout(0.5)(merge)
    x = BatchNormalization()(x)
    x = Dense(600, activation='elu')(x)

    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(600, activation='elu')(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model = Model(inputs=[seq1, seq2,leaks_input], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=1024)
#skf=KFold(y.shape[0],n_folds=5,shuffle=True,random_state=1024)
te_pred=np.zeros(X_train_1.shape[0])
test_pred=np.zeros((X_test_1.shape[0],1))
cnt=0
score=0
for idx_train, idx_val in skf:
    X_train_1_tr=X_train_1[idx_train]
    X_train_1_te=X_train_1[idx_val]
    X_train_2_tr=X_train_2[idx_train]
    X_train_2_te=X_train_2[idx_val]
    leaks_tr=leaks[idx_train]
    leaks_te=leaks[idx_val]
    y_tr=y[idx_train]
    y_te=y[idx_val]

    model = build_model()
    early_stop = EarlyStopping(patience=2)
    check_point = ModelCheckpoint('paipaidai.hdf5', monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    history = model.fit([X_train_1_tr,X_train_2_tr,leaks_tr], y_tr, batch_size = 1024, epochs = 10,validation_data=([X_train_1_te,X_train_2_te,leaks_te], y_te),callbacks=[early_stop,check_point])
    model.load_weights('paipaidai.hdf5')
    preds_te = model.predict([X_train_1_te,X_train_2_te,leaks_te])
    te_pred[idx_val] = preds_te[:, 0]
    #print(y_te.shape)
    #print(preds_te.shape)

    #print("!!!##########################!!!score_test:",log_loss(y_te,preds_te))
    #score+=log_loss(y_te,preds_te)
    preds = model.predict([X_test_1,X_test_2,test_leaks])
    test_pred+=preds
    #break


#score/=5
score=log_loss(y,te_pred)
print(score)
name="plantsgo_%s"%str(round(score,6))
print(score)
t_p = pd.DataFrame()
t_p[name]=te_pred
t_p.to_csv("../meta_features/%s_train.csv"%name,index=False)

test_pred/=5
sub = pd.DataFrame()
sub[name]=test_pred[:,0]
sub.to_csv("../meta_features/%s_test.csv"%name,index=False)
