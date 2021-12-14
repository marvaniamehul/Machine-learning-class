
# coding: utf-8

# In[2]:


from sklearn.datasets import fetch_mldata
import numpy as np
import time
import collections
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
MAX_ITERATIONS = 120


# In[3]:


#X_train, X_test, label_train, label_test = train_test_split(nolrarray, nist.target, test_size=0.10, random_state=42) 


# In[4]:


def shouldStop(oldcentroids, centroids, iterations):
    if iterations > MAX_ITERATIONS: return True
    count = 0;
    if (np.array_equal(centroids,oldcentroids)):
        return True
    return False


# In[5]:


def getLabels(dataset,centroids):

    hashmap = {}
    distmatrix=euclidean_distances(dataset,centroids)

    #print(eucdis.shape)
    for i in range(dataset.shape[0]):
        index = np.argmin(distmatrix[i],axis=0)
        if index in hashmap:
            hashmap[index].append(i)
        else:
            list1 = [i]
            hashmap[index] = list1
    
    return hashmap


# In[6]:


def getCentroids(dataset, labels, k, iterations):
    new_centroids = []
    for centroid in labels.keys():
        tempa = np.mean(dataset[labels[centroid]],axis=0)
        new_centroids.append(tempa)
        
    if(len(new_centroids) < k):
        temindex = np.random.randint(dataset.shape[0], size=(k - len(new_centroids)))
        for index in temindex:
            new_centroids.append(dataset[index])
    
    
   
    return np.reshape(new_centroids,(k,dataset.shape[1]))
        


# In[7]:


import operator
def kmeans(dataSet, k):
    
    idx = np.random.randint(dataSet.shape[0], size=k)
  
    centroids = dataSet[idx]

    iterations = 0
    oldCentroids = np.zeros(shape=(k,dataSet.shape[1]))
   
    
    while not shouldStop(oldCentroids, centroids, iterations):
        print("Running: " + str(iterations))
        oldCentroids = centroids
        iterations += 1
        labels = getLabels(dataSet, centroids)
        centroids = getCentroids(dataSet, labels, k, iterations)
        
    print("Done")
    return centroids
           


# In[8]:


def calculateEvaluationmatrix(hashmap, givenlabels,data):
    maxclus = []
    hasmap = {}
    ginindex = []
    MJ = []
    for key in hashmap.keys():
        temp = {}
        for datapoint in hashmap[key]:

            templabel = givenlabels[datapoint]
            if templabel in temp:
                temp[templabel] += 1
            else:
                temp[templabel] = 1
        datapercen = float(0)
        for keytemp in temp.keys():
            datapercen += ((temp[keytemp]/float(len(hashmap[key]))) ** 2)

        ginindex.append(float(1)-float(datapercen))
        MJ.append(float(len(hashmap[key])))
        maxclus.append(max(temp.items(), key=operator.itemgetter(1))[1])
    
    summation = 0
    for a in maxclus:
        summation += a

    ginisum = float(0)
    for i in range(len(ginindex)):
        ginisum += ginindex[i] * MJ[i]


    print("Purity: " + str(summation/data.shape[0]))

    print("Giniindex: " + str(ginisum/data.shape[0]))


# In[9]:


nist = fetch_mldata('MNIST original')
mnist = nist.data
mnistnew = mnist/255;
print(type(mnistnew))
print("Data Loaded")
start_time = time.time()
final = kmeans(np.array(mnistnew),10)
hashmap = getLabels(mnistnew, final)
givenlabels = nist.target
calculateEvaluationmatrix(hashmap,givenlabels,np.array(mnistnew))


# In[25]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
newsgroups_train = fetch_20newsgroups(subset="train",remove={'headers','quotes','footers'})

newsgroups_traindata = newsgroups_train.data
vectorizer = TfidfVectorizer(analyzer='word',stop_words="english",max_df=0.95, min_df = 0)
vectors = vectorizer.fit_transform(newsgroups_traindata)

print(type(vectors))
print(vectors.shape)
print("Data Loaded")
final2 = kmeans(vectors,20)
hasmap = getLabels(vectors, final2)
newsgroups_labels = newsgroups_train.target
calculateEvaluationmatrix(hasmap,newsgroups_labels,vectors)


# In[27]:


final2 = kmeans(vectors,10)
hasmap = getLabels(vectors, final2)
newsgroups_labels = newsgroups_train.target
calculateEvaluationmatrix(hasmap,newsgroups_labels,vectors)


# In[29]:


final2 = kmeans(vectors,40)
hasmap = getLabels(vectors, final2)
newsgroups_labels = newsgroups_train.target
calculateEvaluationmatrix(hasmap,newsgroups_labels,vectors)


# In[30]:


from utils import mnist_reader
train_data, train_labels = mnist_reader.load_mnist('data/', kind='train')
mnistnew = train_data/255;
print(type(mnistnew))
print("Data Loaded")
start_time = time.time()
final = kmeans(np.array(mnistnew),20)
hashmap = getLabels(mnistnew, final)
calculateEvaluationmatrix(hashmap,train_labels,np.array(mnistnew))


# In[31]:


final = kmeans(np.array(mnistnew),10)
hashmap = getLabels(mnistnew, final)
calculateEvaluationmatrix(hashmap,train_labels,np.array(mnistnew))


# In[33]:


final = kmeans(np.array(mnistnew),5)
hashmap = getLabels(mnistnew, final)
calculateEvaluationmatrix(hashmap,train_labels,np.array(mnistnew))

