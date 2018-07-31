import pandas as pd

import warnings
import numpy as np
warnings.filterwarnings(action='ignore')




#编辑距离
def edit(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]

#最长公共子序列
def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)
    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    """Following steps build L[m+1][n+1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]

def word_edit_share(row):
    q1words = []
    q2words = []
    for word in str(row['words_x']).lower().split():
        q1words.append(word)
    for word in str(row['words_y']).lower().split():
        q2words.append(word)
    R = edit(q1words,q2words)
    return R
def word_edit_share_char(row):
    q1words = []
    q2words = []
    for word in str(row['chars_x']).lower().split():
        q1words.append(word)
    for word in str(row['chars_y']).lower().split():
        q2words.append(word)
    R = edit(q1words,q2words)
    return R

def word_lcs_share(row):
    q1words = []
    q2words = []
    for word in str(row['words_x']).lower().split():
        q1words.append(word)
    for word in str(row['words_y']).lower().split():
        q2words.append(word)
    R = lcs(q1words,q2words)
    return R
def word_lcs_share_char(row):
    q1words = []
    q2words = []
    for word in str(row['chars_x']).lower().split():
        q1words.append(word)
    for word in str(row['chars_y']).lower().split():
        q2words.append(word)
    R = lcs(q1words,q2words)
    return R

if __name__ == '__main__':
    df_train = pd.read_csv('data/x_train.csv')
    df_test = pd.read_csv('data/x_test.csv')

    # df_train.rename(columns={'words_x': 'question1', 'words_y': 'question2'}, inplace=True)
    # df_test.rename(columns={'words_x': 'question1', 'words_y': 'question2'}, inplace=True)


    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    x_train['edit_dis'] = df_train.apply(word_edit_share, axis=1, raw=True)
    x_test['edit_dis'] = df_test.apply(word_edit_share, axis=1, raw=True)

    x_train['edit_dis_char'] = df_train.apply(word_edit_share_char, axis=1, raw=True)
    x_test['edit_dis_char'] = df_test.apply(word_edit_share_char, axis=1, raw=True)

    x_train['lcs_dis'] = df_train.apply(word_lcs_share, axis=1, raw=True)
    x_test['lcs_dis'] = df_test.apply(word_lcs_share, axis=1, raw=True)

    x_train['lcs_dis_char'] = df_train.apply(word_lcs_share_char, axis=1, raw=True)
    x_test['lcs_dis_char'] = df_test.apply(word_lcs_share_char, axis=1, raw=True)

    print(x_train)
    x_train.to_csv('feature/feature_lcs_train.csv', index=False)
    x_test.to_csv('feature/feature_lcs_test.csv', index=False)
