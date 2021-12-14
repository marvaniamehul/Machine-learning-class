
# coding: utf-8

# In[19]:


import math
import numpy as np
text_file = open("/Users/mmarvania/Documents/DM/gaussian.txt", "r");
lines = text_file.readlines();


# In[20]:


matrix = []
count = 0
for line in lines:
    values = line.split(" ")
    locallist = []
    locallist.append([float(values[0])])
    locallist.append([float(values[1])])
    matrix.append(locallist)
    


# In[21]:


matrix = np.array(matrix)
mean = np.array([[[1],[1]],[[1],[1]]])
variance = np.array([[[2,0],[0,3]],[[3,0],[0,5]]])
N = 6000
M = 2
d = 2
pivalues = np.array([0.3,0.7])
prevlogliklihood = 0.000000000
flag = 1


# In[22]:


def EStep(mean, variance, matrix, N, M, d, pivalues):
    dummy = []
    global prevlogliklihood
    global flag
    responsibility = []
    constant = float((math.pi * 2) ** (-d/float(2)))
    for i in range(N):
        
        x = matrix[i]
        temp = []
        totalvalue = float(0)
       
        for m in range(M):
            meu = mean[m]
            
            deter = (np.linalg.det(variance[m])) ** float(-1/2)
            matrix1 = np.subtract(x, meu)
            matrix1T = matrix1.transpose()
            matrix2 = np.linalg.inv(variance[m])
            matrix3 = np.matmul(matrix1T,matrix2)
            finalmatrix = np.linalg.det(np.matmul(matrix3,matrix1))
            expval = np.exp((-1 * finalmatrix)/2)
        
            temp.append(float(float(expval) * float(deter) * float(constant)))
            totalvalue += ((expval * deter * constant) * pivalues[m])
        
        dummy.append(temp)
        ztemp = []
        for j in range(M):
            ztemp.append(pivalues[j] * (float((dummy[i][j])/totalvalue)))
        responsibility.append(ztemp)
    sumtemp = float(0)
    responsibility = np.array(responsibility)
    sumtemp = float(0)
    log_likelihood = np.sum(np.log(np.sum(responsibility.T, axis = 1)))
    print(log_likelihood)
    
    if round(log_likelihood,3) == round(prevlogliklihood,3):
        flag = 0
    
    prevlogliklihood = log_likelihood
    return responsibility
        
def gaussianmix(mean, variance, matrix, N, M, d, pivalues):
    p = 1
    while(1):
        print("Iterations: " + str(p))
        p+=1
     
        responsibility = EStep(mean, variance, matrix, N, M, d, pivalues)

        varianceN = []
        meannew = []
        pivaluesnew = []
        for m in range(M):

            
            num = np.array([[float(0),float(0)],[float(0),float(0)]])
            ressum = float(0)
            meannewsum = np.array([[float(0)],[float(0)]])
            for i in range(N):
                meannewsum += (matrix[i] * responsibility[i][m])
                ressum += responsibility[i][m]
            meannew.append(meannewsum/float(ressum))
            
            meu = meannew[m]
            
            for i in range(N):
                x = matrix[i]
                matrixtemp = np.subtract(x, meu)
                transposeM = matrixtemp.transpose()
                matrix3 = np.matmul(matrixtemp,transposeM) * responsibility[i][m]
                num += matrix3
                
            
            
            varianceN.append(num/float(ressum))
           
            pivaluesnew.append(ressum/N)

        
        varianceN = np.array(varianceN)
        meannew = np.array(meannew)
        pivaluesnew = np.array(pivaluesnew)
      
    
        if flag == 0:
            print(varianceN)
            print(meannew)
            print(pivaluesnew)
            break
            
        
        mean = meannew
        variance = varianceN
        pivalues = pivaluesnew  


# In[10]:


gaussianmix(mean, variance, matrix, N, M, d, pivalues)


# In[11]:


text_file = open("/Users/mmarvania/Documents/DM/3gaussian.txt", "r");
lines = text_file.readlines();
matrix = []
count = 0
for line in lines:
    values = line.split(" ")
    locallist = []
    locallist.append([float(values[0])])
    locallist.append([float(values[1])])
    matrix.append(locallist)
    
matrix = np.array(matrix)
mean = np.array([[[1],[1]],[[1],[1]],[[1],[1]]])
variance = np.array([[[1,0],[0,2]],[[2,0],[0,1]],[[1,0],[0,3]]])
N = 10000
M = 3
d = 2
pivalues = np.array([0.3,0.3,0.4])
prevlogliklihood = 0.000000000
flag = 1

gaussianmix(mean, variance, matrix, N, M, d, pivalues)


# In[46]:


from sklearn.mixture import GaussianMixture
from utils import mnist_reader
import collections
train_data, train_labels = mnist_reader.load_mnist('data/', kind='train')
fashmnistnew = train_data/255;
fashmnistnewnp = np.array(fashmnistnew)

gmmresultfsh = GaussianMixture(n_components=10, covariance_type='diag').fit(fashmnistnewnp)
print(gmmresultfsh.converged_)
print(gmmresultfsh.covariances_.shape)
print(gmmresultfsh.means_.shape)
print(gmmresultfsh.weights_)
    


# In[36]:


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


# In[37]:


hashmap = {}
datapoint = 0
import operator
for i in gmmresultfsh.predict(fashmnistnewnp):
    if i in hashmap:
        hashmap[i].append(datapoint)
    else:
        list1 = [datapoint]
        hashmap[i] = list1[:]
    datapoint += 1

calculateEvaluationmatrix(hashmap,train_labels,fashmnistnewnp )


# In[45]:


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
    
    labels.append(locallist[len(locallist) - 1])
    del locallist[-1]
    matrix.append(locallist)

gmmresultfsh = GaussianMixture(n_components=10, covariance_type='diag').fit(matrix)
print(gmmresultfsh.converged_)
print(gmmresultfsh.covariances_.shape)
print(gmmresultfsh.means_.shape)
print(gmmresultfsh.weights_)


hashmap = {}
datapoint = 0
for i in gmmresultfsh.predict(matrix):
    if i in hashmap:
        hashmap[i].append(datapoint)
    else:
        list1 = [datapoint]
        hashmap[i] = list1[:]
    datapoint += 1

calculateEvaluationmatrix(hashmap,labels,np.array(matrix))

