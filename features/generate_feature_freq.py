


from collections import defaultdict

import numpy as np
import pandas as pd

df_train = pd.read_csv('data/x_train.csv')
df_test = pd.read_csv('data/x_test.csv')

df_train.rename(columns={'words_x': 'question1', 'words_y': 'question2'}, inplace=True)
df_test.rename(columns={'words_x': 'question1', 'words_y': 'question2'}, inplace=True)

ques = pd.concat([df_train[['question1', 'question2']],
                  df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')
q_dict = defaultdict(set)
for i in range(ques.shape[0]):
    q_dict[ques.question1[i]].add(ques.question2[i])
    q_dict[ques.question2[i]].add(ques.question1[i])

def q1_freq(row):
    return(len(q_dict[row['question1']]))

def q2_freq(row):
    return( len(q_dict[row['question2']]))

def q1_q2_intersect(row):
    return( len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))


df_train['q1_q2_intersect'] = df_train.apply(q1_q2_intersect, axis=1, raw=True)
df_train['q1_freq'] = df_train.apply(q1_freq, axis=1, raw=True)
df_train['q2_freq'] = df_train.apply(q2_freq, axis=1, raw=True)

df_test['q1_q2_intersect'] = df_test.apply(q1_q2_intersect, axis=1, raw=True)
df_test['q1_freq'] = df_test.apply(q1_freq, axis=1, raw=True)
df_test['q2_freq'] = df_test.apply(q2_freq, axis=1, raw=True)


df_train=df_train.drop(['label', 'question1','question2','chars_x','chars_y'], axis=1)
df_test=df_test.drop(['label', 'question1','question2','chars_x','chars_y'], axis=1)

print(df_train)
print(df_test)
df_train.to_csv('feature/feature_freq_train.csv', index=False)
df_test.to_csv('feature/feature_freq_test.csv', index=False)