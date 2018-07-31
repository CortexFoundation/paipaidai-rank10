import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

stop_words = ["你","好"]

def wmd(s1, s2):
    s1 = str(s1).split()
    s2 = str(s2).split()

    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).split()
    words = [w for w in words if not w in stop_words]
    #words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def build_features(data):
    X = pd.DataFrame()
    X['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
    question1_vectors = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data.question1.values)):
        question1_vectors[i, :] = sent2vec(q)

    question2_vectors  = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data.question2.values)):
        question2_vectors[i, :] = sent2vec(q)
    #
    # X['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                           np.nan_to_num(question2_vectors))]
    #
    # X['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                           np.nan_to_num(question2_vectors))]
    #
    # X['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                           np.nan_to_num(question2_vectors))]
    #
    # X['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                           np.nan_to_num(question2_vectors))]
    #
    # X['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                           np.nan_to_num(question2_vectors))]
    #
    # X['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                           np.nan_to_num(question2_vectors))]
    #
    # X['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                           np.nan_to_num(question2_vectors))]

    X['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    X['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    X['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    X['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]


    return X

if __name__ == '__main__':
    df_train = pd.read_csv('data/x_train.csv')
    df_test = pd.read_csv('data/x_test.csv')

    df_train.rename(columns={'chars_x': 'question1', 'chars_y': 'question2'}, inplace=True)
    df_test.rename(columns={'chars_x': 'question1', 'chars_y': 'question2'}, inplace=True)

    model = gensim.models.KeyedVectors.load_word2vec_format('data/char_embed_vec.txt', binary=False)

    X_train=build_features(df_train)
    X_test = build_features(df_test)
    print(X_train)
    path = "feature/"
    X_train.to_csv('feature/feature_cnn1_train.csv', index=False)
    X_test.to_csv('feature/feature_cnn1_test.csv', index=False)


