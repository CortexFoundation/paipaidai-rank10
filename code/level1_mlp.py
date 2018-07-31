# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import time
from scipy import sparse
import os
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

#############################################################################################################################
#MAGIC_FEATURE
def get_train_features(features):
    X_train=pd.DataFrame()
    for featurename in features:
        fea=pd.read_csv("../generate_feature/feature/"+featurename)
        X_train = pd.concat((X_train, fea), axis=1)
    return X_train

train_features = [
    "feature_cnn_trian.csv",
    "feature_freq_train.csv",
    "feature_fuzz_train.csv",
    "feature_lcs_train.csv",
    "feature_lcs_sim_train.csv",
    "feature_word_match_trian.csv",
    "feature_cnn1_train.csv",
    "feature_cnn2_train.csv",
    "feature_token_train.csv",
    "tfidf_sim_l2_train.csv",
    "tfidf_sim_l2_bigram_train.csv",
    "train_conn_feat.csv",
    "train_degree_feat.csv"
]

test_features = [
    "feature_cnn_test.csv",
    "feature_freq_test.csv",
    "feature_fuzz_test.csv",
    "feature_lcs_test.csv",
    "feature_lcs_sim_test.csv",
    "feature_word_match_test.csv",
    "feature_cnn1_test.csv",
    "feature_cnn2_test.csv",
    "feature_token_test.csv",
    "tfidf_sim_l2_test.csv",
    "tfidf_sim_l2_bigram_test.csv",
    "test_conn_feat.csv",
    "test_degree_feat.csv"
]

leaks = get_train_features(train_features).values
test_leaks = get_train_features(test_features).values

ss = StandardScaler()
ss.fit(np.vstack((leaks, test_leaks)))
leaks = ss.transform(leaks)
test_leaks = ss.transform(test_leaks)
#############################################################################################################################

###################################################################################################################################
#get data
file_path = "../generate_feature/model/"
name_list = []
for i in os.listdir(file_path):
    if "train" in i:
        name_list.append(i)

train_tfidf = [pd.read_pickle(file_path + i)[:] for i in name_list]
test_tfidf = [pd.read_pickle(file_path + i.replace("train", "test"))[:] for i in name_list]

X_train = sparse.hstack(train_tfidf).tocsr()
X_test = sparse.hstack(test_tfidf).tocsr()

train = pd.read_csv('../input/train.csv')
y=train["label"]
print(y.value_counts())
###################################################################################################################################
# 建立模型
# Define the model
from model_zoo import *
from sklearn.cross_validation import StratifiedKFold,KFold

skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=1024)
#skf=KFold(y.shape[0],n_folds=5,shuffle=True,random_state=1024)
te_pred=np.zeros(X_train.shape[0])
test_pred=np.zeros((X_test.shape[0],1))
cnt=0
score=0
for idx_train, idx_val in skf:
    X_train_tr=X_train[idx_train]
    X_train_te=X_train[idx_val]

    leaks_tr=leaks[idx_train]
    leaks_te=leaks[idx_val]
    y_tr=y[idx_train]
    y_te=y[idx_val]

    model = mlp(X_train.shape[1],leaks_tr.shape[1])
    early_stop = EarlyStopping(patience=2)
    check_point = ModelCheckpoint('paipaidai.hdf5', monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    history = model.fit([X_train_tr,leaks_tr], y_tr, batch_size = 1024, epochs = 100,validation_data=([X_train_te,leaks_te], y_te),callbacks=[early_stop,check_point])
    model.load_weights('paipaidai.hdf5')
    preds_te = model.predict([X_train_te,leaks_te])
    te_pred[idx_val] = preds_te[:, 0]
    #print(y_te.shape)
    #print(preds_te.shape)

    #print("!!!##########################!!!score_test:",log_loss(y_te,preds_te))
    #score+=log_loss(y_te,preds_te)
    preds = model.predict([X_test,test_leaks])
    test_pred+=preds
    #break


#score/=5
score=log_loss(y,te_pred)
print(score)
name="mlp_%s"%str(round(score,6))
print(score)
t_p = pd.DataFrame()
t_p[name]=te_pred
t_p.to_csv("../meta_features/%s_train.csv"%name,index=False)

test_pred/=5
sub = pd.DataFrame()
sub[name]=test_pred[:,0]
sub.to_csv("../meta_features/%s_test.csv"%name,index=False)
