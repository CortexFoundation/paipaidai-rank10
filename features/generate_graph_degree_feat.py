import pandas as pd
import networkx as nx

train_df = pd.read_csv('data/train.csv')
del train_df["label"]
train_df.columns=['question1', 'question2']
test_df = pd.read_csv('data/test.csv')
test_df.columns=['question1', 'question2']

df = pd.concat([train_df, test_df])

g = nx.Graph()
g.add_nodes_from(df.question1)
g.add_nodes_from(df.question2)
edges = list(df[['question1', 'question2']].to_records(index=False))
g.add_edges_from(edges)
# g.remove_edges_from(g.selfloop_edges())


def get_edge_count(g, q):
    return g.degree(q)


def get_edge_count_all(g, row):
    return sum(g.degree([row.question1, row.question2]).values())


def get_edge_count_diff(g, row):
    return g.degree(row.question1) - g.degree(row.question2)


train_df['q1_degree'] = train_df.apply(lambda row: get_edge_count(g, row.question1), axis=1)
train_df['q2_degree'] = train_df.apply(lambda row: get_edge_count(g, row.question2), axis=1)
#train_df['qall_degree'] = train_df.apply(lambda row: get_edge_count_all(g, row), axis=1)


test_df['q1_degree'] = test_df.apply(lambda row: get_edge_count(g, row.question1), axis=1)
test_df['q2_degree'] = test_df.apply(lambda row: get_edge_count(g, row.question2), axis=1)
#test_df['qall_degree'] = test_df.apply(lambda row: get_edge_count_all(g, row), axis=1)

comb_tr = train_df[['q1_degree', 'q2_degree']]
comb_te = test_df[['q1_degree', 'q2_degree']]


comb_tr.to_csv('feature/train_degree_feat.csv',index=None)
comb_te.to_csv('feature/test_degree_feat.csv',index=None)

