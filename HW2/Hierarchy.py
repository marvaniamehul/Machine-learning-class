
# coding: utf-8

# In[15]:


from utils import mnist_reader
import numpy as np
#t10K
train_data, train_labels = mnist_reader.load_mnist('data/', kind='t10K')
mnistnew = train_data/255;
print(mnistnew.shape)


# In[16]:


from scipy.cluster.hierarchy import dendrogram, linkage


# In[18]:


Z = linkage(mnistnew, 'ward')


# In[19]:


import scipy
X = scipy.cluster.hierarchy.fcluster(Z,10,criterion='maxclust')


# In[20]:


len(X)


# In[21]:


hashmap1 = {}
for i in range(len(X)):
    if X[i] not in hashmap1:
        hashmap1[X[i]] = [i]
    else:
        hashmap1[X[i]].append(i)


# In[25]:


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


# In[26]:


import operator
calculateEvaluationmatrix(hashmap1,train_labels,mnistnew)

