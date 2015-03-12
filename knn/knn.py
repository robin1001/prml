from numpy import *

def knn(test, train, train_label, k):
    n1, n2 = test.shape[0], train.shape[0]
    ans= [None]*n1
    for i in range(n1):
        diff = train - tile(test[i, :], (n2, 1))
        dist = sum(diff**2, 1) #sqrt 
        near_index = dist.argsort() 
        dic = {}
        for j in range(k):
            label = train_label[near_index[j]]
            dic[label] = dic.get(label, 0) + 1
        sort_dic = sorted(dic.iteritems(), key=lambda x: x[1], reverse=True)
        ans[i] = sort_dic[0][0]
    return ans


if __name__ == '__main__':
    train = random.rand(10, 2)
    train[:5,:] = train[:5,:] + 2
    label = ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
    test = random.rand(5, 2)
    test[:3,:] = test[:3,:]+2
    print knn(test, train, label, 1)
