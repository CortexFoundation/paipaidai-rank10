import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
df_train = pd.read_csv('data/train_bigram.csv')
df_test = pd.read_csv('data/test_bigram.csv')

# df_train.rename(columns={'words_x': 'question1', 'words_y': 'question2'}, inplace=True)
# df_test.rename(columns={'words_x': 'question1', 'words_y': 'question2'}, inplace=True)
# df_train['word1_bigram'] = df_train['word1_bigram'].apply(lambda x: str(x).split())
# df_train['word2_bigram'] = df_train['word2_bigram'].apply(lambda x: str(x).split())
# df_test['word1_bigram'] = df_test['word1_bigram'].apply(lambda x: str(x).split())
# df_test['word2_bigram'] = df_test['word2_bigram'].apply(lambda x: str(x).split())

data_all =df_train
max_features = None
ngram_range = (1,2)
min_df = 3
print('Generate tfidf')
feats= ['word1_bigram','word2_bigram']
vect_orig = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=min_df)

corpus = []
for f in feats:
    data_all[f] = data_all[f].astype(str)
    corpus+=data_all[f].values.tolist()
#questions = list(data_all['question1']) + list(data_all['question2'])
#print(questions)
print(corpus[:10])
vect_orig.fit(corpus)
path="model/"

for f in feats:
    train_tfidf = vect_orig.transform(df_train[f].astype(str).values.tolist())
    test_tfidf = vect_orig.transform(df_test[f].astype(str).values.tolist())
    pd.to_pickle(train_tfidf, path + 'train_%s_tfidf_v2.pkl' % f)
    pd.to_pickle(test_tfidf, path + 'test_%s_tfidf_v2.pkl' % f)


print('Generate chars tfidf')
feats_char= ['char1_bigram','char2_bigram']

# df_train['char1_bigram'] = df_train['char1_bigram'].apply(lambda x: str(x).split())
# df_train['char2_bigram'] = df_train['char2_bigram'].apply(lambda x: str(x).split())
# df_test['char1_bigram'] = df_test['char1_bigram'].apply(lambda x: str(x).split())
# df_test['char2_bigram'] = df_test['char2_bigram'].apply(lambda x: str(x).split())
vect_orig1 = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=min_df)

corpus1 = []
for f in feats_char:
    data_all[f] = data_all[f].astype(str)
    corpus1+=data_all[f].values.tolist()

vect_orig1.fit(
    corpus1
    )

for f in feats_char:
    train_tfidf = vect_orig1.transform(df_train[f].astype(str).values.tolist())
    test_tfidf = vect_orig1.transform(df_test[f].astype(str).values.tolist())
    pd.to_pickle(train_tfidf,path+'train_%s_tfidf_v2.pkl'%f)
    pd.to_pickle(test_tfidf,path+'test_%s_tfidf_v2.pkl'%f)