import pandas as pd
import networkx as nx
from networkx.algorithms import approximation as approx

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


def conn2(g, row):
    try:
        ret = approx.node_connectivity(g, row.question1, row.question2)
    except:
        return 1
    return ret


def get_edge_count(g, q):
    return g.degree(q)


def get_edge_count_all(g, row):
    return sum(g.degree([row.question1, row.question2]).values())


def get_edge_count_diff(g, row):
    return g.degree(row.question1) - g.degree(row.question2)


train_df['conn2'] = train_df.apply(lambda row: conn2(g, row), axis=1)
test_df['conn2'] = test_df.apply(lambda row: conn2(g, row), axis=1)

comb_tr = train_df[['conn2']]
comb_te = test_df[['conn2']]
#
#
comb_tr.to_csv('feature/train_conn_feat.csv',index=None)
comb_te.to_csv('feature/test_conn_feat.csv',index=None)

