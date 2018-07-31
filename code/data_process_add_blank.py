import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
ques=pd.read_csv('../input/question.csv')
ques.columns=["q1","w1","c1"]
train=train.merge(ques,on="q1",how="left")
test=test.merge(ques,on="q1",how="left")
ques.columns=["q2","w2","c2"]
train=train.merge(ques,on="q2",how="left")
test=test.merge(ques,on="q2",how="left")

def add_blank(a,b):
    c=[]
    for i in a.split(" "):
        if i not in b:
            c.append(i)
        else:
            c.append("QAQ")
    return " ".join(c)

train["w1_new"]=list(map(lambda a,b:add_blank(a,b),train["w1"],train["w2"]))
train["w2_new"]=list(map(lambda a,b:add_blank(a,b),train["w2"],train["w1"]))

test["w1_new"]=list(map(lambda a,b:add_blank(a,b),test["w1"],train["w2"]))
test["w2_new"]=list(map(lambda a,b:add_blank(a,b),test["w2"],train["w1"]))

train[["label","w1_new","w2_new"]].to_csv("../input/train_blank_words.csv",index=None)
test[["w1_new","w2_new"]].to_csv("../input/test_blank_words.csv",index=None)
