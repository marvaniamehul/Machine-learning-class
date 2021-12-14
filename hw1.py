from sklearn.datasets import fetch_mldata
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import tempfile
from sklearn.model_selection import train_test_split
custom_data_home = tempfile.mkdtemp()


# nist = fetch_mldata('MNIST original')

# mnist = nist.data
# print(type(mnist))
# print("Data Loaded")
# mnist.astype(float)

# nolrarray = mnist/float(255)

# X_train, X_test, label_train, label_test = train_test_split(nolrarray, nist.target, test_size=0.10, random_state=42)

# print("Shape for Mnist Trained Data: " + str(X_train.shape))
# print("Shape for Mnist Test Data: " + str(X_test.shape))
# start_time = time.time()
# cosinersltmnist = cosine_similarity(nolrarray)
# print("Cosine shape for MNIST: " + str(cosinersltmnist.shape))
# print("Cosine for MNIST Time Taken {} secs".format(time.time() - start_time))
# start_time = time.time()
# euclideanrsltecld = euclidean_distances(nolrarray)
# print("Euclidean shape for MNIST: " + str(euclideanrsltecld.shape))
# print("Euclidean for MNIST Time Taken {} secs".format(time.time() - start_time))

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
newsgroups_traindata = newsgroups_train.data

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_traindata)
start_time = time.time()
#cosinerslt20 = cosine_similarity(vectors)
#print("Cosine for 20NewsGroup Time Taken {} secs".format(time.time() - start_time))
#print("Cosine Shape for 20NewsGroup: " + str(cosinerslt20.shape))
#print("Time Taken {} secs".format(time.time() - start_time))
train_labelsnews = newsgroups_train.target
test_labelsnews = newsgroups_test.target

start_time = time.time()
#euclidean20 = euclidean_distances(vectors)
#print("Euclidean for 20NewsGroup Time Taken {} secs".format(time.time() - start_time))
#print("Euclidean Shape for 20NewsGroup: " + str(euclidean20.shape))
#print("Time Taken {} secs".format(time.time() - start_time))

### KNN Algorithm.
vocab = vectorizer.vocabulary_

newvectorizer = TfidfVectorizer(vocabulary=vocab)
vectorstest = newvectorizer.fit_transform(newsgroups_test.data)

print("Shape for Newsgroup20 Trained Data: " + str(vectors.shape))
print("Shape for Newsgroup20 Test Data: " + str(vectorstest.shape))

def find_k_nearest_neighbours(training_matrix, testing_matrix , idx, k, type1, train_labelslocal, test_labelslocal):

	cosinerslt20knn = cosine_similarity(testing_matrix, training_matrix)
	print("cosine shape")
	print(cosinerslt20knn.shape)
	# print("Cosine between the testing and training")
	# print(cosinerslt20knn.shape)

	testid = 0
	count = 0
	for outerdoc in cosinerslt20knn:
		flag = 0
		# print("Document: " + str(test_labelslocal[testid]) + "\n")
		indexes = np.argpartition(outerdoc,-k)[-k:]
		# print("Possibilities:\n")
		hashmap = {}
		for i in range(k):
			if train_labelslocal[indexes[i]] in hashmap:
				hashmap[train_labelslocal[indexes[i]]] += 1
			else:
				hashmap[train_labelslocal[indexes[i]]] = 1

		mmmax = 1
		labelcurr = 0
		
		for key, value in hashmap.items():
			if value > mmmax:
				labelcurr = key
				mmmax = value
		# print("***********")
		

		if(test_labelslocal[testid] == labelcurr):
			count += 1

		if(testid == idx):
			print(type1 + " of index " + str(idx) + " has label: " + str(test_labelslocal[testid]))
			print("Nearest " + type1 + " with above "+ type1 +" has label: " + str(labelcurr))

		testid += 1

	#get vab of trainibng data
	#assign vacab of

	print("Accuracy: " + str((count/testid)*100) + "\n")

# print(cosine_similarity(vectorstest[:10]))
find_k_nearest_neighbours(vectors, vectorstest,2, 5, "Document", train_labelsnews, test_labelsnews)
#find_k_nearest_neighbours(X_train, X_test, 6464, 5, "Image", label_train, label_test)


