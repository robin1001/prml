import math
import os
import re

#trainning
def train(hamset, spamset):          
    wordset = dict() #key, value:(hn sn p(w|h) p(w|n))
    ham_sum, spam_sum = 0, 0
    for item in hamset:
        if item not in wordset:
            wordset[item] = [1, 1, 0, 0]
            ham_sum += 1
        wordset[item][0] += 1
        ham_sum += 1

    for item in spamset:
        if item not in wordset:
            wordset[item] = [1, 1, 0, 0]
            spam_sum += 1
        wordset[item][1] += 1
        spam_sum += 1

    for item in wordset:
        wordset[item][2] = math.log(float(wordset[item][0])/ham_sum)
        wordset[item][3] = math.log(float(wordset[item][1])/spam_sum)

    total = float(ham_sum + spam_sum)
    return ham_sum/total, spam_sum/total, wordset

def test(params, x):
    p_ham, p_spam, wordset = params
    ham, spam = math.log(p_ham), math.log(p_spam)
    for item in x:
        if item in wordset:
            ham += wordset[item][2]
            spam += wordset[item][3]
    return 1 if ham > spam  else 0 

def get_tokens(strs):
    pattern = re.compile('\\W+')
    tokens = pattern.split(strs)

    return [tok.lower() for tok in tokens if len(tok) > 1]

def load_data():
    ham_dir, spam_dir = './train/ham', './train/spam'
    ham_data, spam_data = [], []
    for f in os.listdir(ham_dir):
        path = os.path.join(ham_dir, f)
        with open(path) as fid:
            ham_data.extend(get_tokens(fid.read()))

    for f in os.listdir(spam_dir):
        path = os.path.join(spam_dir, f)
        with open(path) as fid:
            spam_data.extend(get_tokens(fid.read()))
    return ham_data, spam_data

if __name__ == '__main__':
    ham_data, spam_data = load_data()
    params = train(ham_data, spam_data)

    test_dir = './test'
    for f in os.listdir(test_dir):
        path = os.path.join(test_dir, f)
        with open(path) as fid:
            res = test(params, get_tokens(fid.read()))
            if res > 0: 
                print "%s:%s" % (path, "ham")
            else:   
                print "%s:%s" % (path, "spam")
