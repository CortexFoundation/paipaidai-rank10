import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
import functools
def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in row['words_x']:
        q1words[word] = 1
    for word in row['words_y']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R
def char_match_share(row):
    q1words = {}
    q2words = {}
    for word in row['chars_x']:
        q1words[word] = 1
    for word in row['chars_y']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R
def jaccard_word(row):
    wic = set(row['words_x']).intersection(set(row['words_y']))
    uw = set(row['words_x']).union(row['words_y'])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))
def jaccard_char(row):
    wic = set(row['chars_x']).intersection(set(row['chars_y']))
    uw = set(row['chars_x']).union(row['chars_y'])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))
def wc_diff(row):
    return abs(len(row['words_x']) - len(row['words_y']))
def char_diff(row):
    return abs(len(row['chars_x']) - len(row['chars_y']))
def wc_ratio(row):
    l1 = len(row['words_x'])*1.0
    l2 = len(row['words_y'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2
def char_ratio(row):
    l1 = len(row['chars_x']) * 1.0
    l2 = len(row['chars_y'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2
def wc_diff_unique(row):
    return abs(len(set(row['words_x'])) - len(set(row['words_y'])))
def char_diff_unique(row):
    return abs(len(set(row['chars_x'])) - len(set(row['chars_y'])))
def wc_ratio_unique(row):
    l1 = len(set(row['words_x'])) * 1.0
    l2 = len(set(row['words_y']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2
def char_ratio_unique(row):
    l1 = len(set(row['chars_x'])) * 1.0
    l2 = len(set(row['chars_y']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)
def load_char_weight(data):
    train_qs = pd.Series(data['chars_x'].tolist() + data['chars_y'].tolist())
    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    return weights
def load_word_weight(data):
    train_qs = pd.Series(data['words_x'].tolist() + data['words_y'].tolist())
    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    return weights
def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['words_x']:
        q1words[word] = 1
    for word in row['words_y']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def tfidf_char_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['chars_x']:
        q1words[word] = 1
    for word in row['chars_y']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def build_features(data):
    X = pd.DataFrame()
    X['word_match'] = data.apply(word_match_share, axis=1, raw=True)   # 1
    X['char_match'] = data.apply(char_match_share, axis=1, raw=True)   # 2
    X['jaccard_word'] = data.apply(jaccard_word, axis=1, raw=True)     # 3
    X['jaccard_char'] = data.apply(jaccard_char, axis=1, raw=True)     # 4
    X['wc_diff'] = data.apply(wc_diff, axis=1, raw=True)               # 5
    X['char_diff'] = data.apply(char_diff, axis=1, raw=True)           # 6
    X['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True)             # 7
    X['char_ratio'] = data.apply(char_ratio, axis=1, raw=True)        # 8
    X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True)   # 9
    X['char_diff_unique'] = data.apply(char_diff_unique, axis=1, raw=True) # 10
    X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True)   # 11
    X['char_ratio_unique'] = data.apply(char_ratio_unique, axis=1, raw=True)  # 12

    f = functools.partial(tfidf_word_match_share, weights=load_word_weight(data))
    X['tfidf_wm'] = data.apply(f, axis=1, raw=True)                          #13
    d = functools.partial(tfidf_char_match_share, weights=load_char_weight(data))
    X['tfidf_charm'] = data.apply(d, axis=1, raw=True)                     #14

    return X
if __name__ == '__main__':
    df_train = pd.read_csv('data/x_train.csv')
    df_test = pd.read_csv('data/x_test.csv')
    df_train['words_x'] = df_train['words_x'].map(lambda x: str(x).split())
    df_train['words_y'] = df_train['words_y'].map(lambda x: str(x).split())
    df_train['chars_x'] = df_train['chars_x'].map(lambda x: str(x).split())
    df_train['chars_y'] = df_train['chars_y'].map(lambda x: str(x).split())

    df_test['words_x'] = df_test['words_x'].map(lambda x: str(x).split())
    df_test['words_y'] = df_test['words_y'].map(lambda x: str(x).split())
    df_test['chars_x'] = df_test['chars_x'].map(lambda x: str(x).split())
    df_test['chars_y'] = df_test['chars_y'].map(lambda x: str(x).split())

    print('Building Features')
    X_train = build_features(df_train)

    X_test = build_features(df_test)

    X_train.to_csv("feature/feature_word_match_trian.csv", index=False)
    X_test.to_csv("feature/feature_word_match_test.csv", index=False)
