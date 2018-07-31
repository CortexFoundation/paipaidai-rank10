import numpy as np
import pandas as pd

df_train =  pd.read_csv('data/train.csv')
df_test =  pd.read_csv('data/test.csv')

test_label = pd.read_csv('lgb_xgb_demo.csv')
test_label=test_label.values[:,0]

from collections import defaultdict

REPEAT = 2 #a reasonable number which can consider your updates iteratively but not ruin the predictions

DUP_THRESHOLD = 0.5 #classification threshold for duplicates
NOT_DUP_THRESHOLD = 0.1 #classification threshold for non-duplicates
#Since the data is unbalanced, our mean prediction is around 0.16. So this is the reason of unbalanced thresholds

MAX_UPDATE = 0.2 # maximum update on the dup probability (a high choice may ruin the predictions)
DUP_UPPER_BOUND = 0.98 # do not update dup probabilities above this threshold
NOT_DUP_LOWER_BOUND = 0.01 # do not update dup probabilities below this threshold

for i in range(REPEAT):
    dup_neighbors = defaultdict(set)

    for dup, q1, q2 in zip(df_train["label"], df_train["q1"], df_train["q2"]):
        if dup:
            dup_neighbors[q1].add(q2)
            dup_neighbors[q2].add(q1)

    for dup, q1, q2 in zip(test_label, df_test["q1"], df_test["q2"]):
        if dup > DUP_THRESHOLD:
            dup_neighbors[q1].add(q2)
            dup_neighbors[q2].add(q1)

    count = 0
    for index, (q1, q2) in enumerate(zip(df_test["q1"], df_test["q2"])):
        dup_neighbor_count = len(dup_neighbors[q1].intersection(dup_neighbors[q2]))
        if dup_neighbor_count > 0 and test_label[index] < DUP_UPPER_BOUND:
            update = min(MAX_UPDATE, (DUP_UPPER_BOUND - test_label[index]) / 2)
            test_label[index] += update
            count += 1

    print("Edited:", count)

for j in range(REPEAT):
    not_dup_neighbors = defaultdict(set)

    for dup, q1, q2 in zip(df_train["label"], df_train["q1"], df_train["q2"]):
        if not dup:
            not_dup_neighbors[q1].add(q2)
            not_dup_neighbors[q2].add(q1)

    for dup, q1, q2 in zip(test_label, df_test["q1"], df_test["q2"]):
        if dup < NOT_DUP_THRESHOLD:
            not_dup_neighbors[q1].add(q2)
            not_dup_neighbors[q2].add(q1)

    count = 0
    for index, (q1, q2) in enumerate(zip(df_test["q1"], df_test["q2"])):
        dup_neighbor_count = len(not_dup_neighbors[q1].intersection(not_dup_neighbors[q2]))
        if dup_neighbor_count > 0 and test_label[index] > NOT_DUP_LOWER_BOUND:
            update = min(MAX_UPDATE, (test_label[index] - NOT_DUP_LOWER_BOUND) / 2)
            test_label[index] -= update
            count += 1

    print("Edited:", count)

sub = pd.DataFrame()
sub["y_pre"] =test_label
sub.to_csv("edit_lgb_xgb_demo.csv", index=False)