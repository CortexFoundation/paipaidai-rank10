import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
seed = 1024
np.random.seed(seed)
path = "model/"

def calc_cosine_dist(text_a ,text_b):
    # print(text_a)
    # print(text_b)
    # print("---------------------------------")
    #y=pairwise_distances(text_a, text_b, metric='l1')[0][0]
    y = pairwise_distances(text_a, text_b, metric='l2')[0][0]
    #y= cosine_similarity(text_a, text_b)[0][0]
    # print(y)
    # print(y1)
    # print("------------------------------------------------")
    return y



print('Tfidf word Similarity part')

train_question1_tfidf = pd.read_pickle(path+'train_words_x_tfidf_v2.pkl')
test_question1_tfidf = pd.read_pickle(path+'test_words_x_tfidf_v2.pkl')
train_question2_tfidf = pd.read_pickle(path+'train_words_y_tfidf_v2.pkl')
test_question2_tfidf = pd.read_pickle(path+'test_words_y_tfidf_v2.pkl')

train_tfidf_sim = []
for r1,r2 in zip(train_question1_tfidf,train_question2_tfidf):
    train_tfidf_sim.append(calc_cosine_dist(r1,r2))
test_tfidf_sim = []
for r1,r2 in zip(test_question1_tfidf,test_question2_tfidf):
    test_tfidf_sim.append(calc_cosine_dist(r1,r2))
train_tfidf_sim = np.array(train_tfidf_sim)
test_tfidf_sim = np.array(test_tfidf_sim)
X = pd.DataFrame()
Y=pd.DataFrame()
X["tfidf_word_sim"]=train_tfidf_sim
Y["tfidf_word_sim"]=test_tfidf_sim

print(X)


del train_question1_tfidf
del test_question1_tfidf
del train_question2_tfidf
del test_question2_tfidf

print('Tfidf char Similarity part')
train_question1_bigram_tfidf = pd.read_pickle(path+'train_chars_x_tfidf_v2.pkl')
test_question1_bigram_tfidf = pd.read_pickle(path+'test_chars_x_tfidf_v2.pkl')
train_question2_bigram_tfidf = pd.read_pickle(path+'train_chars_y_tfidf_v2.pkl')
test_question2_bigram_tfidf = pd.read_pickle(path+'test_chars_y_tfidf_v2.pkl')

train_bigram_tfidf_sim = []
for r1,r2 in zip(train_question1_bigram_tfidf,train_question2_bigram_tfidf):
    train_bigram_tfidf_sim.append(calc_cosine_dist(r1,r2))
test_bigram_tfidf_sim = []
for r1,r2 in zip(test_question1_bigram_tfidf,test_question2_bigram_tfidf):
    test_bigram_tfidf_sim.append(calc_cosine_dist(r1,r2))
train_bigram_tfidf_sim = np.array(train_bigram_tfidf_sim)
test_bigram_tfidf_sim = np.array(test_bigram_tfidf_sim)
# pd.to_pickle(train_bigram_tfidf_sim,path+"train_bigram_tfidf_sim.pkl")
# pd.to_pickle(test_bigram_tfidf_sim,path+"test_bigram_tfidf_sim.pkl")
X["tfidf_char_sim"]=train_bigram_tfidf_sim
Y["tfidf_char_sim"]=test_bigram_tfidf_sim
del train_question1_bigram_tfidf
del test_question1_bigram_tfidf
del train_question2_bigram_tfidf
del test_question2_bigram_tfidf
X.to_csv('feature/tfidf_sim_l2_train.csv', index=False)
Y.to_csv('feature/tfidf_sim_l2_test.csv', index=False)