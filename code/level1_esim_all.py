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
emb_char_dic={}
with open("../input/char_embed.txt") as f:
    word_emb=f.readlines()
    word_emb=word_emb
    print(len(word_emb))
    for w in word_emb:
        w=w.replace("\n","")
        content=w.split(" ")
        emb_char_dic[content[0].lower()]=np.array(content[1:])

emb_word_dic={}
with open("../input/word_embed.txt") as f:
    word_emb=f.readlines()
    word_emb=word_emb
    print(len(word_emb))
    for w in word_emb:
        w=w.replace("\n","")
        content=w.split(" ")
        emb_word_dic[content[0].lower()]=np.array(content[1:])

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
MAX_SEQUENCE_LENGTH_CHAR = 40
MAX_SEQUENCE_LENGTH_WORD = 20
MAX_NB_WORDS_CHAR = 5000
MAX_NB_WORDS_WORD = 50000
EMBEDDING_DIM = 300
DROPOUT = 0.1
#process data
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS_CHAR,)
tokenizer.fit_on_texts(list(train["c1"])+list(test["c1"])+list(train["c2"])+list(test["c2"]))
column="c1"
sequences_all = tokenizer.texts_to_sequences(list(train[column]))
sequences_test = tokenizer.texts_to_sequences(list(test[column]))
X_train_1_char = pad_sequences(sequences_all, maxlen=MAX_SEQUENCE_LENGTH_CHAR,padding='post')
X_test_1_char = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH_CHAR,padding='post')
column="c2"
sequences_all = tokenizer.texts_to_sequences(list(train[column]))
sequences_test = tokenizer.texts_to_sequences(list(test[column]))
X_train_2_char = pad_sequences(sequences_all, maxlen=MAX_SEQUENCE_LENGTH_CHAR,padding='post')
X_test_2_char = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH_CHAR,padding='post')

word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS_CHAR, len(word_index))+1
print(nb_words)

ss=0
word_embedding_matrix_char = np.zeros((nb_words, EMBEDDING_DIM))
print(len(word_index.items()))
for word, i in word_index.items():
    if word in emb_char_dic.keys():
        ss+=1
        word_embedding_matrix_char[i] = emb_char_dic[word]
    else:
        pass


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS_WORD,)
tokenizer.fit_on_texts(list(train["w1"])+list(test["w1"])+list(train["w2"])+list(test["w2"]))
column="w1"
sequences_all = tokenizer.texts_to_sequences(list(train[column]))
sequences_test = tokenizer.texts_to_sequences(list(test[column]))
X_train_1_word = pad_sequences(sequences_all, maxlen=MAX_SEQUENCE_LENGTH_WORD,padding='post')
X_test_1_word = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH_WORD,padding='post')
column="w2"
sequences_all = tokenizer.texts_to_sequences(list(train[column]))
sequences_test = tokenizer.texts_to_sequences(list(test[column]))
X_train_2_word = pad_sequences(sequences_all, maxlen=MAX_SEQUENCE_LENGTH_WORD,padding='post')
X_test_2_word = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH_WORD,padding='post')

word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS_WORD, len(word_index))+1
print(nb_words)

ss=0
word_embedding_matrix_word = np.zeros((nb_words, EMBEDDING_DIM))
print(len(word_index.items()))
for word, i in word_index.items():
    if word in emb_word_dic.keys():
        ss+=1
        word_embedding_matrix_word[i] = emb_word_dic[word]
    else:
        pass




y=train["label"]
print(y.value_counts())
###################################################################################################################################
# 建立模型
# Define the model
from esim_all import esim
from sklearn.cross_validation import StratifiedKFold,KFold

skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=1024)
#skf=KFold(y.shape[0],n_folds=5,shuffle=True,random_state=1024)
te_pred=np.zeros(X_train_1_char.shape[0])
test_pred=np.zeros((X_test_1_char.shape[0],1))
cnt=0
score=0
for idx_train, idx_val in skf:
    X_train_1_char_tr=X_train_1_char[idx_train]
    X_train_1_char_te=X_train_1_char[idx_val]
    X_train_2_char_tr=X_train_2_char[idx_train]
    X_train_2_char_te=X_train_2_char[idx_val]

    X_train_1_word_tr=X_train_1_word[idx_train]
    X_train_1_word_te=X_train_1_word[idx_val]
    X_train_2_word_tr=X_train_2_word[idx_train]
    X_train_2_word_te=X_train_2_word[idx_val]

    leaks_tr=leaks[idx_train]
    leaks_te=leaks[idx_val]
    y_tr=y[idx_train]
    y_te=y[idx_val]

    model = esim(word_embedding_matrix_char,word_embedding_matrix_word,leaks.shape[1])
    early_stop = EarlyStopping(patience=2)
    check_point = ModelCheckpoint('paipaidai.hdf5', monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    history = model.fit([X_train_1_char_tr,X_train_2_char_tr,X_train_1_word_tr,X_train_2_word_tr,leaks_tr], y_tr, batch_size = 512, epochs = 10,
                        validation_data=([X_train_1_char_te,X_train_2_char_te,X_train_1_word_te,X_train_2_word_te,leaks_te], y_te),
                        callbacks=[early_stop,check_point])
    model.load_weights('paipaidai.hdf5')
    preds_te = model.predict([X_train_1_char_te,X_train_2_char_te,X_train_1_word_te,X_train_2_word_te,leaks_te])
    te_pred[idx_val] = preds_te[:, 0]
    #print(y_te.shape)
    #print(preds_te.shape)

    #print("!!!##########################!!!score_test:",log_loss(y_te,preds_te))
    #score+=log_loss(y_te,preds_te)
    preds = model.predict([X_test_1_char,X_test_2_char,X_test_1_word,X_test_2_word,test_leaks])
    test_pred+=preds
    #break


#score/=5
score=log_loss(y,te_pred)
print(score)
name="esim_all_%s"%str(round(score,6))
print(score)
t_p = pd.DataFrame()
t_p[name]=te_pred
t_p.to_csv("../meta_features/%s_train.csv"%name,index=False)

test_pred/=5
sub = pd.DataFrame()
sub[name]=test_pred[:,0]
sub.to_csv("../meta_features/%s_test.csv"%name,index=False)
