#encoding=utf8
import pandas as pd
import lightgbm as lgb
import re
import time
import numpy as np
import math
import gc
import pickle
import os
from sklearn.metrics import roc_auc_score,log_loss
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
#from com_util import *
import lightgbm
from sklearn.cross_validation import StratifiedKFold,KFold

def stacking(clf,train_x,train_y,test_x,clf_name,class_num=1):
    #predictors=list(train_x.columns)
    #train_x=train_x.values
    #test_x=test_x.values
    folds = 5
    seed = 2018
    kf = StratifiedKFold(train_y,n_folds=5,shuffle=True,random_state=1024)
    #kf = KFold(train_y.shape[0],n_folds=5,shuffle=True,random_state=1024)

    train=np.zeros((train_x.shape[0],class_num))
    test=np.zeros((test_x.shape[0],class_num))
    test_pre=np.zeros((folds,test_x.shape[0],class_num))
    cv_scores=[]

    for i,(train_index,test_index) in enumerate(kf):
        tr_x=train_x[train_index]
        tr_y=train_y[train_index]
        te_x=train_x[test_index]
        te_y = train_y[test_index]

        train_matrix = clf.Dataset(tr_x, label=tr_y)
        test_matrix = clf.Dataset(te_x, label=te_y)

        params = {
                  'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': 'binary_logloss',
                  'min_child_weight': 1.5,
                  'num_leaves': 2**5,
                  'lambda_l2': 10,
                  'subsample': 0.7,
                  'colsample_bytree': 0.5,
                  'colsample_bylevel': 0.5,
                  'learning_rate': 0.1,
                  'seed': 2018,
                  'nthread': 16,
                  'silent': True,
                  }


        num_round = 5000
        early_stopping_rounds = 100
        if test_matrix:
            model = clf.train(params, train_matrix,num_round,valid_sets=[test_matrix,test_matrix],
                              early_stopping_rounds=early_stopping_rounds
                              )
            #print("\n".join(("%s: %.2f" % x) for x in
                            #sorted(zip(predictors, model.feature_importance("gain")), key=lambda x: x[1],
                                   #reverse=True)))
            pre= model.predict(te_x,num_iteration=model.best_iteration).reshape((te_x.shape[0],1))
            train[test_index]=pre
            test_pre[i, :]= model.predict(test_x, num_iteration=model.best_iteration).reshape((test_x.shape[0],1))
            cv_scores.append(log_loss(te_y, pre))

        print("%s now score is:"%clf_name,cv_scores)
    test[:]=test_pre.mean(axis=0)
    print("%s_score_list:"%clf_name,cv_scores)
    print("%s_score_mean:"%clf_name,np.mean(cv_scores))
    with open("score_cv.txt", "a") as f:
        f.write("%s now score is:" % clf_name + str(cv_scores) + "\n")
        f.write("%s_score_mean:"%clf_name+str(np.mean(cv_scores))+"\n")
    return train.reshape(-1),test.reshape(-1),np.mean(cv_scores)


def lgb_oof(x_train, y_train, x_valid):
    lgb_train, lgb_test,cv_scores = stacking(lightgbm, x_train, y_train, x_valid,"lgb")
    return lgb_train, lgb_test,cv_scores

def get_train_features(features):
    X_train=pd.DataFrame()
    for featurename in features:
        fea=pd.read_csv("../generate_feature/feature/"+featurename)
        X_train = pd.concat((X_train, fea), axis=1)
    return X_train

def get_data():
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

    X_train = get_train_features(train_features)
    X_test = get_train_features(test_features)

    file_path="../generate_feature/model/"
    name_list=[]
    for i in os.listdir(file_path):
        if "train" in i:
            name_list.append(i)

    train_tfidf = [pd.read_pickle(file_path+i)[:] for i in name_list]
    test_tfidf = [pd.read_pickle(file_path+i.replace("train","test"))[:] for i in name_list]

    X_train=sparse.hstack([X_train]+train_tfidf).tocsr()
    X_test=sparse.hstack([X_test]+test_tfidf).tocsr()


    df_train = pd.read_csv('../input/train.csv')
    y_train = df_train['label']
    return X_train,y_train,X_test

def main():
    train_x,train_y,test_x=get_data()

    lgb_train, lgb_test, m = lgb_oof(train_x, train_y, test_x)

    sub=pd.DataFrame(lgb_train)
    sub.columns=["lgb_%.6f"%m]
    sub.to_csv("../meta_features/lgb_%.6f_train"%m,index=None)

    sub=pd.DataFrame(lgb_test)
    sub.columns=["lgb_%.6f"%m]
    sub.to_csv("../meta_features/lgb_%.6f_test"%m,index=None)

if __name__=="__main__":
    main()
