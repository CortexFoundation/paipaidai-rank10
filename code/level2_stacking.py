#encoding=utf8
import os
from sklearn.cross_validation import KFold,StratifiedKFold
import pandas as pd
import numpy as np
from scipy import sparse
import xgboost
import lightgbm
import catboost


from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss,mean_absolute_error,mean_squared_error,roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB


##############################################################分类####################################################
def stacking(clf,train_x,train_y,test_x,clf_name,class_num=1):
    train=np.zeros((train_x.shape[0],class_num))
    test=np.zeros((test_x.shape[0],class_num))
    test_pre=np.empty((folds,test_x.shape[0],class_num))
    cv_scores=[]
    for i,(train_index,test_index) in enumerate(kf):
        tr_x=train_x[train_index]
        tr_y=train_y[train_index]
        te_x=train_x[test_index]
        te_y = train_y[test_index]
        if clf_name in ["rf","ada","gb","et","lr","knn","mnb","ovr","gnb"]:
            clf.fit(tr_x,tr_y)
            pre=clf.predict_proba(te_x)[:,1].reshape((te_x.shape[0],1))
            train[test_index]=pre
            test_pre[i,:]=clf.predict_proba(test_x)[:,1].reshape((test_x.shape[0],1))
            cv_scores.append(log_loss(te_y, pre))
        elif clf_name in ["lsvc"]:
            clf.fit(tr_x,tr_y)
            pre=clf.decision_function(te_x)
            train[test_index]=pre
            test_pre[i,:]=clf.decision_function(test_x)
            cv_scores.append(log_loss(te_y, pre))
        elif clf_name in ["xgb"]:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, label=te_y, missing=-1)
            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'logloss',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 7,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.02,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 12,
                      "num_class": class_num
                      }

            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'),
                         (test_matrix, 'eval')
                         ]
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round=num_round,evals=watchlist,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre= model.predict(test_matrix,ntree_limit=model.best_ntree_limit).reshape((te_x.shape[0],1))
                train[test_index]=pre
                test_pre[i, :]= model.predict(z, ntree_limit=model.best_ntree_limit).reshape((test_x.shape[0],1))
                print(log_loss(te_y, pre))
                cv_scores.append(log_loss(te_y, pre))
        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            #z = clf.Dataset(test_x, label=te_y)
            #z=test_x
            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'binary_logloss',
                'min_child_weight': 1.5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.5,
                'colsample_bylevel': 0.7,
                'learning_rate': 0.01,
                'seed': 2017,
                'nthread': 12,
                'silent': True,
            }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre= model.predict(te_x,num_iteration=model.best_iteration).reshape((te_x.shape[0],1))
                train[test_index]=pre
                test_pre[i, :]= model.predict(test_x, num_iteration=model.best_iteration).reshape((test_x.shape[0],1))
                cv_scores.append(log_loss(te_y, pre))

        elif clf_name in ["cat"]:
            clf = catboost.CatBoostClassifier(loss_function='Logloss',
                                     eval_metric='AUC',
                                     iterations=5000,
                                     learning_rate=0.02,
                                     depth=6,
                                     rsm=0.7,
                                     od_type='Iter',
                                     od_wait=700,
                                     logging_level='Verbose',
                                     allow_writing_files=False,
                                     metric_period=100,
                                     random_seed=1)
            clf.fit(tr_x,tr_y,eval_set=(te_x,te_y),use_best_model=True)
            pre=clf.predict_proba(te_x)[:,1].reshape((te_x.shape[0],1))
            train[test_index]=pre
            test_pre[i, :]= clf.predict_proba(test_x)[:,1].reshape((test_x.shape[0],1))
            cv_scores.append(log_loss(te_y, pre))

        elif clf_name in ["nn"]:
            from keras.layers import Dense, Dropout, BatchNormalization
            from keras.optimizers import SGD,RMSprop
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from keras.utils import np_utils
            from keras.regularizers import l2
            from keras.models import Sequential
            clf = Sequential()
            clf.add(Dense(1024, input_dim=tr_x.shape[1],activation="relu"))
            #clf.add(SReLU())
            clf.add(Dropout(0.6))
            clf.add(Dense(512,activation="relu"))
            #clf.add(SReLU())
            #clf.add(Dense(64, activation="relu", W_regularizer=l2()))
            clf.add(Dropout(0.3))
            clf.add(Dense(class_num, activation="sigmoid"))
            clf.summary()
            early_stopping = EarlyStopping(monitor='val_loss', patience=20)
            reduce = ReduceLROnPlateau(min_lr=0.0002,factor=0.05)
            clf.compile(optimizer="rmsprop", loss="binary_crossentropy")
            clf.fit(tr_x, tr_y,
                      batch_size=1280,
                      nb_epoch=1000,
                      validation_data=[te_x, te_y],
                      callbacks=[early_stopping#,reduce
                                 ])
            pre=clf.predict_proba(te_x)
            train[test_index]=pre
            test_pre[i,:]=clf.predict_proba(test_x)
            cv_scores.append(log_loss(te_y, pre))
        else:
            raise IOError("Please add new clf.")
        print("%s now score is:"%clf_name,cv_scores)
        with open("score.txt","a") as f:
            f.write("%s now score is:"%clf_name+str(cv_scores)+"\n")
    test[:]=test_pre.mean(axis=0)
    print("%s_score_list:"%clf_name,cv_scores)
    print("%s_score_mean:"%clf_name,np.mean(cv_scores))
    with open("score.txt", "a") as f:
        f.write("%s_score_mean:"%clf_name+str(np.mean(cv_scores))+"\n")
    return train.reshape((-1,1)),test.reshape((-1,1))

def rf(x_train, y_train, x_valid):
    randomforest = RandomForestClassifier(n_estimators=1200, max_depth=16, n_jobs=-1, random_state=2017, max_features="auto",verbose=1)
    rf_train, rf_test = stacking(randomforest, x_train, y_train, x_valid,"rf")
    return rf_train, rf_test,"rf"

def ada(x_train, y_train, x_valid):
    adaboost = AdaBoostClassifier(n_estimators=60, random_state=2017, learning_rate=0.01)
    ada_train, ada_test = stacking(adaboost, x_train, y_train, x_valid,"ada")
    return ada_train, ada_test,"ada"

def gb(x_train, y_train, x_valid):
    gbdt = GradientBoostingClassifier(learning_rate=0.06, n_estimators=100, subsample=0.8, random_state=2017,max_depth=5,verbose=1)
    gbdt_train, gbdt_test = stacking(gbdt, x_train, y_train, x_valid,"gb")
    return gbdt_train, gbdt_test,"gb"

def et(x_train, y_train, x_valid):
    extratree = ExtraTreesClassifier(n_estimators=1200, max_depth=24, max_features="auto", n_jobs=-1, random_state=2017,verbose=1)
    et_train, et_test = stacking(extratree, x_train, y_train, x_valid,"et")
    return et_train, et_test,"et"

def ovr(x_train, y_train, x_valid):
    est=RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=2017, max_features="auto",
                               verbose=1)
    ovr = OneVsRestClassifier(est,n_jobs=-1)
    ovr_train, ovr_test = stacking(ovr, x_train, y_train, x_valid,"ovr")
    return ovr_train, ovr_test,"ovr"

def xgb(x_train, y_train, x_valid):
    xgb_train, xgb_test = stacking(xgboost, x_train, y_train, x_valid,"xgb")
    return xgb_train, xgb_test,"xgb"

def lgb(x_train, y_train, x_valid):
    xgb_train, xgb_test = stacking(lightgbm, x_train, y_train, x_valid,"lgb")
    return xgb_train, xgb_test,"lgb"

def cat(x_train, y_train, x_valid):
    xgb_train, xgb_test = stacking(catboost.CatBoostClassifier, x_train, y_train, x_valid,"cat")
    return xgb_train, xgb_test,"cat"

def gnb(x_train, y_train, x_valid):
    gnb=GaussianNB()
    gnb_train, gnb_test = stacking(gnb, x_train, y_train, x_valid,"gnb")
    return gnb_train, gnb_test,"gnb"

def lr(x_train, y_train, x_valid):
    scale=StandardScaler()
    scale.fit(x_train)
    x_train=scale.transform(x_train)
    x_valid=scale.transform(x_valid)
    logisticregression=LogisticRegression(n_jobs=-1,random_state=2017,C=0.1,max_iter=200)
    lr_train, lr_test = stacking(logisticregression, x_train, y_train, x_valid, "lr")
    return lr_train, lr_test, "lr"

def fm(x_train, y_train, x_valid):
    pass


def lsvc(x_train, y_train, x_valid):
    x_train=np.log10(x_train+1)
    x_valid=np.log10(x_valid+1)

    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = 0
    x_train[where_are_inf] = 0
    where_are_nan = np.isnan(x_valid)
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_nan] = 0
    x_valid[where_are_inf] = 0

    scale=StandardScaler()
    scale.fit(x_train)
    x_train=scale.transform(x_train)
    x_valid=scale.transform(x_valid)

    #linearsvc=SVC(probability=True,kernel="linear",random_state=2017,verbose=1)
    #linearsvc=SVC(probability=True,kernel="linear",random_state=2017,verbose=1)
    linearsvc=LinearSVC(random_state=2017)
    lsvc_train, lsvc_test = stacking(linearsvc, x_train, y_train, x_valid, "lsvc")
    return lsvc_train, lsvc_test, "lsvc"

def knn(x_train, y_train, x_valid):
    scale=StandardScaler()
    scale.fit(x_train)
    x_train=scale.transform(x_train)
    x_valid=scale.transform(x_valid)

    pca = PCA(n_components=10)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_valid = pca.transform(x_valid)
    kneighbors=KNeighborsClassifier(n_neighbors=200,n_jobs=-1)
    knn_train, knn_test = stacking(kneighbors, x_train, y_train, x_valid, "knn")
    return knn_train, knn_test, "knn"

def nn(x_train, y_train, x_valid):
    scale=StandardScaler()
    scale.fit(x_train)
    x_train=scale.transform(x_train)
    x_valid=scale.transform(x_valid)
    nn_train, nn_test = stacking("", x_train, y_train, x_valid, "nn")
    return nn_train, nn_test, "nn"
###########################################################################################################

#####################################################回归##################################################

#####################################################获取数据##############################################

###########################################################################################################
def get_data():
    file_path="../meta_features/"
    name_list=[]
    for i in os.listdir(file_path):
        if ("_train" in i) or ("_x." in i) or ("train_" in i):
            name_list.append(i)
    print(name_list)
    train_x=pd.concat([pd.read_csv(file_path+name) for name in name_list],axis=1).fillna(0)
    test_x=pd.concat([pd.read_csv(file_path+name.replace("_train","_test").replace("_x.","_y.").replace("train_.","test_.")) for name in name_list],axis=1).fillna(0)
    train_x=train_x.values
    test_x=test_x.values


    train_y=pd.read_csv("../input/train.csv")["label"]
    return train_x,train_y,test_x

    return train_x,train_y,test_x

if __name__=="__main__":
    with open("score.txt", "a") as f:
        f.write("stacking_level2" + ":\n")

    np.random.seed(2017)
    x_train, y_train, x_valid = get_data()
 
    folds = 5
    #seed = 2017
    #kf = KFold(x_train.shape[0], n_folds=folds, shuffle=True, random_state=seed)
    kf = StratifiedKFold(y_train,n_folds=5,shuffle=True,random_state=1024)

    #############################################选择模型###############################################
    #
    #
    #
    clf_list = [lr,nn,knn,xgb,gb,rf,et,lgb,ada]
    #clf_list = [lgb]
    #clf_list = [cat]
    #clf_list = [xgb_reg,lgb_reg,nn_reg,lgb,xgb,lr,rf,et,gb,nn,knn]  #添加了magic的
    #clf_list = [xgb_reg,lgb_reg,nn_reg,et_reg,rf_reg,lr_reg,ada_reg,gb_reg]   #添加了magic的,补充三个reg
    #
    #
    #
    column_list = []
    train_data_list=[]
    test_data_list=[]
    for clf in clf_list:
        train_data,test_data,clf_name=clf(x_train,y_train,x_valid)
        train_data_list.append(train_data)
        test_data_list.append(test_data)
        column_list.append("%s" % clf_name)

    train = np.concatenate(train_data_list, axis=1)
    test = np.concatenate(test_data_list, axis=1)

    result=test.copy()

    train = pd.DataFrame(train)
    train.columns = column_list

    test = pd.DataFrame(test)
    test.columns = column_list

    train.to_csv("level2_train.csv", index=None)
    test.to_csv("level2_test.csv", index=None)

