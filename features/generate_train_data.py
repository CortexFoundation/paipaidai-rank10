import pandas as pd
import numpy as np

def get_ids(qids):
    ids = []
    for t_ in qids:
        ids.append(int(t_[1:]))
    return np.asarray(ids)
def get_train_texts():
    train_df=pd.DataFrame()
    qes = pd.read_csv("data/question.csv")
    file = pd.read_csv("data/train.csv")
    q1id, q2id = file['q1'], file['q2']
    label = file["label"].values
    id1s, id2s = get_ids(q1id), get_ids(q2id)

    all_words = qes['words']
    all_chars= qes['chars']
    q1 = []
    q2 = []
    c1=[]
    c2=[]
    for t_ in zip(id1s, id2s):
        q1.append(all_words[t_[0]])
        q2.append(all_words[t_[1]])
        c1.append(all_chars[t_[0]])
        c2.append(all_chars[t_[1]])

    train_df["label"]=label
    train_df["words_x"] = q1
    train_df["words_y"] = q2
    train_df["chars_x"] = c1
    train_df["chars_y"] = c2
    return train_df


def get_test_texts():
    train_df = pd.DataFrame()
    qes = pd.read_csv("data/question.csv")
    file = pd.read_csv("data/test.csv")
    file = file.reindex(columns=["label", "q1", "q2"], fill_value=1)
    q1id, q2id = file['q1'], file['q2']
    label = file["label"].values
    id1s, id2s = get_ids(q1id), get_ids(q2id)

    all_words = qes['words']
    all_chars = qes['chars']
    q1 = []
    q2 = []
    c1 = []
    c2 = []
    for t_ in zip(id1s, id2s):
        q1.append(all_words[t_[0]])
        q2.append(all_words[t_[1]])
        c1.append(all_chars[t_[0]])
        c2.append(all_chars[t_[1]])
    train_df["label"] = label
    train_df["words_x"] = q1
    train_df["words_y"] = q2
    train_df["chars_x"] = c1
    train_df["chars_y"] = c2
    return train_df


if __name__ == '__main__':
    path="data/"
    train_x=get_train_texts()
    test_x = get_test_texts()
    train_x.to_csv(path + 'x_train.csv', index=False)
    test_x.to_csv(path + 'x_test.csv', index=False)