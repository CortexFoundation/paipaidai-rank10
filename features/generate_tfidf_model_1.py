import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
df_train = pd.read_csv('data/x_train.csv')
df_test = pd.read_csv('data/x_test.csv')

# df_train.rename(columns={'words_x': 'question1', 'words_y': 'question2'}, inplace=True)
# df_test.rename(columns={'words_x': 'question1', 'words_y': 'question2'}, inplace=True)
# df_train['words_x'] = df_train['words_x'].apply(lambda x: x.split())
# df_train['words_y'] = df_train['words_y'].apply(lambda x: x.split())
# df_test['words_x'] = df_test['words_x'].apply(lambda x: x.split())
# df_test['words_y'] = df_test['words_y'].apply(lambda x: x.split())

data_all =df_train
max_features = None
ngram_range = (1,2)
min_df = 3
print('Generate tfidf')
feats= ['words_x','words_y']
vect_orig = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=min_df)

corpus = []
for f in feats:
    data_all[f] = data_all[f].astype(str)
    corpus+=data_all[f].values.tolist()
print(corpus[:10])
vect_orig.fit(corpus)
path="model/"

for f in feats:
    train_tfidf = vect_orig.transform(df_train[f].astype(str).values.tolist())
    test_tfidf = vect_orig.transform(df_test[f].astype(str).values.tolist())
    pd.to_pickle(train_tfidf, path + 'train_%s_tfidf_v2.pkl' % f)
    pd.to_pickle(test_tfidf, path + 'test_%s_tfidf_v2.pkl' % f)


print('Generate chars tfidf')
feats_char= ['chars_x','chars_y']

# df_train['chars_x'] = df_train['chars_x'].apply(lambda x: x.split())
# df_train['chars_y'] = df_train['chars_y'].apply(lambda x: x.split())
# df_test['chars_x'] = df_test['chars_x'].apply(lambda x: x.split())
# df_test['chars_y'] = df_test['chars_y'].apply(lambda x: x.split())
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