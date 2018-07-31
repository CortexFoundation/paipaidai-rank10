
from fuzzywuzzy import fuzz
import pandas as pd



SAFE_DIV = 0.0001
STOP_WORDS = ["你好"]

def get_token_features(q1, q2):
    token_features = [0.0]*8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[3] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[4] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[5] = int(q1_tokens[0] == q2_tokens[0])
    token_features[6] = abs(len(q1_tokens) - len(q2_tokens))
    token_features[7] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features

def extract_features(df):

    print("token features...")
    token_features = df.apply(lambda x: get_token_features(x["words_x"], x["words_y"]), axis=1)
    df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    df["cwc_max"]       = list(map(lambda x: x[1], token_features))
    df["ctc_min"]       = list(map(lambda x: x[2], token_features))
    df["ctc_max"]       = list(map(lambda x: x[3], token_features))
    df["last_word_eq"]  = list(map(lambda x: x[4], token_features))
    df["first_word_eq"] = list(map(lambda x: x[5], token_features))
    df["abs_len_diff"]  = list(map(lambda x: x[6], token_features))
    df["mean_len"]      = list(map(lambda x: x[7], token_features))

    char_token_features = df.apply(lambda x: get_token_features(x["chars_x"], x["chars_y"]), axis=1)
    df["char_cwc_min"] = list(map(lambda x: x[0], char_token_features))
    df["char_cwc_max"] = list(map(lambda x: x[1], char_token_features))
    df["char_ctc_min"] = list(map(lambda x: x[2], char_token_features))
    df["char_ctc_max"] = list(map(lambda x: x[3], char_token_features))
    df["char_last_word_eq"] = list(map(lambda x: x[4], char_token_features))
    df["char_first_word_eq"] = list(map(lambda x: x[5], char_token_features))
    df["char_abs_len_diff"] = list(map(lambda x: x[6], char_token_features))
    df["char_mean_len"] = list(map(lambda x: x[7], char_token_features))
    return df

if __name__ == '__main__':

    df_train = pd.read_csv('data/x_train.csv')
    df_test = pd.read_csv('data/x_test.csv')

    X_train = extract_features(df_train)
    X_test = extract_features(df_test)

    X_train = df_train.drop(['label', 'words_x', 'words_y', 'chars_x', 'chars_y'], axis=1)
    X_test = df_test.drop(['label', 'words_x', 'words_y', 'chars_x', 'chars_y'], axis=1)

    X_train.to_csv("feature/feature_token_train.csv", index=False)
    X_test.to_csv("feature/feature_token_test.csv", index=False)