
from fuzzywuzzy import fuzz
import pandas as pd


def extract_features(df):
    X = pd.DataFrame()
    print("fuzzy features..")
    X["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["words_x"], x["words_y"]), axis=1)
    X["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["words_x"], x["words_y"]), axis=1)
    X["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["words_x"], x["words_y"]), axis=1)
    X["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["words_x"], x["words_y"]), axis=1)

    print("fuzzy features..char,")
    X["token_set_ratio_char"] = df.apply(lambda x: fuzz.token_set_ratio(x["chars_x"], x["chars_y"]), axis=1)
    X["token_sort_ratio_char"] = df.apply(lambda x: fuzz.token_sort_ratio(x["chars_x"], x["chars_y"]), axis=1)
    X["fuzz_ratio_char"] = df.apply(lambda x: fuzz.QRatio(x["chars_x"], x["chars_y"]), axis=1)
    X["fuzz_partial_ratio_char"] = df.apply(lambda x: fuzz.partial_ratio(x["chars_x"], x["chars_y"]), axis=1)
    return X


if __name__ == '__main__':

    df_train = pd.read_csv('data/x_train.csv')
    df_test = pd.read_csv('data/x_test.csv')

    print('Building Features')
    X_train = extract_features(df_train)

    X_test = extract_features(df_test)

    X_train.to_csv("feature/feature_fuzz_train.csv", index=False)
    X_test.to_csv("feature/feature_fuzz_test.csv", index=False)