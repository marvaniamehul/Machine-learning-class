

from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
import time
import collections
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from sklearn.preprocessing import StandardScaler
import operator
from custom import calculateEvaluationmatrixnew

nist = fetch_mldata('MNIST original')
mnist = nist.data
mnistnew = mnist/255;
X_train, X_test, label_train, label_test = train_test_split(mnistnew,nist.target , test_size=0.09, random_state=42)

data = X_train

meanvector = np.mean(data,axis = 0)
stdmid = np.zeros((784,784))
for i in range(data.shape[0]):
    temp = (data[i] - meanvector).reshape(784,1)
    a = np.matmul(temp,temp.transpose())
    stdmid += a
stdmid = np.array(stdmid)
print(stdmid.shape)
print(data.shape[0])

stdmidNor = stdmid/(data.shape[0] -1)
a = StandardScaler().fit_transform(X_train)
u,s,v = svds(stdmidNor,5)
print(u.shape)

# stdmidNor = stdmid/(data.shape[0] - 1)
# eigenvals, eigvecs = np.linalg.eigh(u)
# print(eigvecs)
# matrix_w = np.hstack((eigvecs[0].reshape(784,1),eigvecs[1].reshape(784,1)))
# for i in range(2,5):
#     matrix_w = np.hstack((matrix_w,eigvecs[i].reshape(784,1)))
# print(matrix_w.shape)

Ytrain = np.matmul(X_train,u)
Ytest = np.matmul(X_test,u)
print(Ytrain.shape) 
print(Ytest.shape)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LResult = LR.fit(Ytrain,label_train)

LResult.score(Ytest,label_test)

u,s,v = svds(stdmidNor,20)
Ytrain = X_train.dot(u)
Ytest = X_test.dot(u)
print(Ytrain.shape)
print(Ytest.shape)

LResult = LR.fit(Ytrain,label_train)
LResult.score(Ytest,label_test)

from sklearn.decomposition import PCA
pca5 = PCA(n_components=5)

resultpca5 = pca5.fit(X_train)

Ytrain = X_train.dot(resultpca5.components_.transpose())
Ytest = X_test.dot(resultpca5.components_.transpose())
LResult = LR.fit(Ytrain,label_train)
LResult.score(Ytest,label_test)

pca10 = PCA(n_components=20)
resultpca10 = pca10.fit(X_train)
Ytrain = X_train.dot(resultpca10.components_.transpose())
Ytest = X_test.dot(resultpca10.components_.transpose())
LResult = LR.fit(Ytrain,label_train)
LResult.score(Ytest,label_test)

text_file = open("/Users/mmarvania/Documents/DM/HW2/spambase.txt", "r");
lines = text_file.readlines();
matrix = []
count = 0
labels = []
for line in lines:
    values = line.split(",")
    locallist = []
    for val in values:
        locallist.append(val)
    a = locallist[len(locallist) - 1].split()
    labels.append(a)
    del locallist[-1]
    locallist = list(map(float,locallist))
    matrix.append(locallist)
matrix=np.array(matrix)

X_trainSP, X_testSP, label_trainSP, label_testSP = train_test_split(matrix,np.array(labels) , test_size=0.20, random_state=42)

for i in range(5,57):
    pca = PCA(n_components=i)

    pca.fit(X_trainSP)
    Ytrain = pca.transform(X_trainSP)

    Ytest = pca.transform(X_testSP)

    # resultpca5 = pca5.fit(X_trainSP)
    # Ytrain = np.matmul(X_trainSP,resultpca5.components_.transpose())
    # Ytest = np.matmul(X_testSP,resultpca5.components_.transpose())
    LResult = LR.fit(Ytrain,label_trainSP)
    print(str(i) + ": " + str(LResult.score(Ytest,label_testSP)))

