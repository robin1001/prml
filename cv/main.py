#!/bin/python
from pylab import *
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.lda import LDA
#load data

def test_classifier(x, y, classifiers):
    result = [] 
    for key, classifier in classifiers.iteritems():
        scores = cross_validation.cross_val_score(classifier, x, y, cv=3)
        result.append(mean(scores))
        #print "%s score:%f" % (key, mean(scores))
    return result
def show_result(r):
    print 'knn score %f' % r[0]
    print 'svm score %f' % r[1]
    print 'naive bayes %f' % r[2]

tmp = np.loadtxt('pp5i_train.gr.csv', dtype=np.str, delimiter=',')
train_data = tmp[1:, 1:].astype(np.int).T
tmp = np.loadtxt('pp5i_test.gr.csv', dtype=np.str, delimiter=',')
test_data = tmp[1:, 1:].astype(np.int).T
label = np.loadtxt('pp5i_train_class.txt', dtype=np.str)[1:]

le = preprocessing.LabelEncoder()
le.fit(label)
train_label = le.transform(label)

classifiers = {'knn': neighbors.KNeighborsClassifier(3),
               'svm': svm.SVC(kernel='rbf', C=10),
               'naive bayes': GaussianNB() }
print 'raw data'
r = test_classifier(train_data, train_label, classifiers)
show_result(r)

print 'pca data demension reduction'
mini, maxi = 2, 20 
n_components = arange(mini, maxi)
results = []
for i in range(mini, maxi):
    pca = PCA(n_components=i, whiten=True)
    pca_data = pca.fit(train_data).transform(train_data)
    r = test_classifier(pca_data, train_label, classifiers)
    results.append(r)
results = np.array(results)
pca_results, index = results.max(0), results.argmax(0)
print index + mini

show_result(pca_results)
plt.figure()
plt.plot(n_components, results[:,0], 'r', label='knn scores')
plt.plot(n_components, results[:,1], 'g', label='svm scores')
plt.plot(n_components, results[:,2], 'b', label='naive bayes scores')
plt.xlabel('nb of pca components')
plt.ylabel('pca CV scores')
plt.legend(loc='lower right')

print 'lda data demension reduction'
mini, maxi = 1, 20 
n_components = arange(mini, maxi)
results = []
for i in range(mini, maxi):
    lda = LDA(n_components=i)
    lda_data = lda.fit(train_data, train_label).transform(train_data)
    r = test_classifier(lda_data, train_label, classifiers)
    results.append(r)
results = np.array(results)
lda_results, index = results.max(0), results.argmax(0)
print index + mini
show_result(lda_results)

plt.figure()
plt.plot(n_components, results[:,0], 'r', label='knn scores')
plt.plot(n_components, results[:,1], 'g', label='svm scores')
plt.plot(n_components, results[:,2], 'b', label='naive bayes scores')
plt.xlabel('nb of lda components')
plt.ylabel('lda CV scores')
plt.legend(loc='lower right')
#plt.show()

#predict on testset, pca(n_componet=10), svm
pca = PCA(n_components=i, whiten=True).fit(train_data)
train_pca_data = pca.transform(train_data)
test_pca_data = pca.transform(test_data)
svm = classifiers['knn']
svm.fit(train_pca_data, train_label)
predict = svm.predict(test_pca_data)
print 'best result: pca(10), svm classifier'
print le.inverse_transform(predict)
