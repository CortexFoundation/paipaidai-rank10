from datetime import datetime
from csv import DictReader
import string

from ngram import getBigram
path = 'data/'

string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
string.punctuation.__add__('?')
string.punctuation.__add__('.')
string.punctuation.__add__(',')


def remove_punctuation(x):
    new_line = [ w for w in list(x) if w not in string.punctuation]
    new_line = ''.join(new_line)
    return new_line

def prepare_bigram(path,out):
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('word1_bigram,word2_bigram,char1_bigram,char2_bigram\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')):
            if c%100000==0:
                print('finished',c)
            q1 = remove_punctuation(str(row['words_x']).lower()).split(' ')
            q2 = remove_punctuation(str(row['words_y']).lower()).lower().split(' ')
            q3 = remove_punctuation(str(row['chars_x']).lower()).split(' ')
            q4 = remove_punctuation(str(row['chars_y']).lower()).lower().split(' ')
            q1_bigram = getBigram(q1)
            q2_bigram = getBigram(q2)
            q3_bigram = getBigram(q3)
            q4_bigram = getBigram(q4)
            q1_bigram = ' '.join(q1_bigram)
            q2_bigram = ' '.join(q2_bigram)
            q3_bigram = ' '.join(q3_bigram)
            q4_bigram = ' '.join(q4_bigram)
            outfile.write('%s,%s,%s,%s\n' % (q1_bigram, q2_bigram,q3_bigram,q4_bigram))


            c+=1
        end = datetime.now()
        print('times:',end-start)

prepare_bigram(path+'x_train.csv',path+'train_bigram.csv')

prepare_bigram(path+'x_test.csv',path+'test_bigram.csv')