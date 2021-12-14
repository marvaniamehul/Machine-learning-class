from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_mldata
import operator
import numpy as np
import operator
from sklearn.model_selection import train_test_split
from custom import calculateEvaluationmatrixnew

nist = fetch_mldata('MNIST original')
mnist = nist.data
mnistnew = mnist/255;
mnistlabel = nist.target


X_train, X_test, label_train, label_test = train_test_split(mnistnew,mnistlabel , test_size=0.10, random_state=42)

LR = LogisticRegression()
print(X_train.shape)

LResult = LR.fit(X_train,label_train)

print(LResult.coef_.shape)
print(LResult.intercept_.shape)
print(LResult.n_iter_.shape)

LResult.score(X_test,label_test)

coefs=LResult.coef_
top_30 = np.argpartition(coefs, -30)[:,range(-30,0)]
top_30.shape

import matplotlib.pyplot as plt
im = plt.imread("0.png")
implot = plt.imshow(im)
plt.scatter(x=top_30[0]/28, y=top_30[0]%28, c='r', s=40)
plt.show()

im = plt.imread("1.png")
implot = plt.imshow(im)
plt.scatter(x=top_30[1]/28, y=top_30[1]%28, c='r', s=40)
plt.show()

im = plt.imread("2.png")
implot = plt.imshow(im)
plt.scatter(x=top_30[2]/28, y=top_30[2]%28, c='r', s=40)
plt.show()

im = plt.imread("3.png")
implot = plt.imshow(im)
plt.scatter(x=top_30[3]/28, y=top_30[3]%28, c='r', s=40)
plt.show()

im = plt.imread("4.png")
implot = plt.imshow(im)
plt.scatter(x=top_30[4]/28, y=top_30[4]%28, c='r', s=40)
plt.show()

im = plt.imread("5.png")
implot = plt.imshow(im)
plt.scatter(x=top_30[5]/28, y=top_30[5]%28, c='r', s=40)
plt.show()

im = plt.imread("6.png")
implot = plt.imshow(im)
plt.scatter(x=top_30[6]/28, y=top_30[6]%28, c='r', s=40)
plt.show()

im = plt.imread("7.png")
implot = plt.imshow(im)
plt.scatter(x=top_30[7]/28, y=top_30[7]%28, c='r', s=40)
plt.show()


im = plt.imread("8.png")
implot = plt.imshow(im)
plt.scatter(x=top_30[8]/28, y=top_30[8]%28, c='r', s=40)
plt.show()

im = plt.imread("9.png")
implot = plt.imshow(im)
plt.scatter(x=top_30[9]/28, y=top_30[9]%28, c='r', s=40)
plt.show()

from sklearn.tree import DecisionTreeClassifier
regr_2 = DecisionTreeClassifier(max_depth=15)

regr_2MN = regr_2.fit(X_train,label_train)

regr_2MN.score(X_test,label_test)

coefs=regr_2MN.tree_.feature
top_30Dec = []

for feat in coefs:
    if(feat > 0):
        top_30Dec.append(feat)
    if(len(top_30Dec) >= 30):
        break
top_30Dec = np.array(list(map(int, top_30Dec)))
plt.scatter(x=top_30Dec/28, y=top_30Dec%28, c='r', s=40)
plt.show()

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

LResultspam = LR.fit(X_trainSP,label_trainSP)
print(LResultspam.score(X_testSP,label_testSP))

coefs=LResult.coef_
top_30 = np.argpartition(coefs, -30)[:,range(-30,0)]
print(coefs.shape)

text_file = open("/Users/mmarvania/Documents/DM/HW3/spamnames.txt", "r");
lines = text_file.readlines();
featurename = []

for line in lines:
    featurename.append(line.split())

top_30 = top_30.reshape((30,))
for a in top_30:
    print(featurename[int(a)])

regr_3 = DecisionTreeClassifier(max_depth=10)
regr_2SP = regr_3.fit(X_testSP,label_testSP)
regr_2SP.score(X_testSP,label_testSP)

coefs=regr_2SP.tree_.feature
print(coefs)
top_30DecSp = []
for feat in coefs:
    if(feat > 0):
        top_30DecSp.append(feat)
    if(len(top_30DecSp) >= 30):
        break
for a in top_30DecSp:
    print(featurename[int(a)])

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
newsgroups_train = fetch_20newsgroups(subset="train")

newsgroups_trainlabel = newsgroups_train.target
vectorizer = TfidfVectorizer()
vectorstrain = vectorizer.fit_transform(newsgroups_train.data)
newsgroups_test = fetch_20newsgroups(subset="test")
vocab = vectorizer.vocabulary_
newvectorizer = TfidfVectorizer(vocabulary=vocab)
vectorstest = newvectorizer.fit_transform(newsgroups_test.data)
newsgroups_testlabel = newsgroups_test.target

print(vectorstrain.shape)

LResult20NG = LR.fit(vectorstrain,newsgroups_trainlabel)

LResultspam.score(vectorstest,newsgroups_testlabel)

feature_names = np.asarray(vectorizer.get_feature_names())
coefs=LResult20NG.coef_
top_30 = np.argpartition(coefs, -30)[:,range(-30,0)]
wordlist = []
for doc in top_30:
    word = []
    for a in doc:
        word.append(feature_names[a])
    wordlist.append(word)
print(wordlist)

regr_20NG = regr_2.fit(vectorstrain,newsgroups_trainlabel)

regr_20NG.score(vectorstest,newsgroups_testlabel)

regr_3 = DecisionTreeClassifier()
regr_20NG = regr_3.fit(vectorstrain,newsgroups_trainlabel)
regr_20NG.score(vectorstest,newsgroups_testlabel)

coefs=regr_20NG.tree_.feature
top_30DecSp = []
for feat in coefs:
    if(feat > 0):
        top_30DecSp.append(feat)
    if(len(top_30DecSp) >= 30):
        break
wordlist = []

for word in top_30DecSp:
    wordlist.append(feature_names[word])
print(wordlist)

