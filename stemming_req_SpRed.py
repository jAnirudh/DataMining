from pandas import read_csv, DataFrame
from bs4 import BeautifulSoup
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import hstack
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from ml_metrics import quadratic_weighted_kappa
from sklearn.cross_validation import cross_val_predict

# create arrays

traindata1 = []
traindata2 = []

#load data
train = read_csv("../train.csv").fillna("")
test  = read_csv("../test.csv").fillna("")
y = train.median_relevance.values
idTestx = test.id.values.astype(int) # isolating the IDs of the Test Data

import numpy
from Levenshtein import jaro_winkler

def score_title(query,title):
    score = 0
    for term in query.lower().split(" "):
        if term in title.lower(): score += 1
    return score

def get_distance(source,destination,distance = jaro_winkler):
    if type(source) == str : return distance(source,destination)
    else: return distance(str(source),str(destination))

trainFeatures = numpy.zeros((len(y),4))
testFeatures  = numpy.zeros((len(idTestx),4))

for i in range(len(y)):
    trainFeatures[i,0] = get_distance(train["query"][i],train["product_title"][i])
    trainFeatures[i,1] = get_distance(train["query"][i],train["product_description"][i])
    trainFeatures[i,2] = len(train["query"][i].split(" "))
    trainFeatures[i,3] = score_title(train["query"][i],train["product_title"][i])

for i in range(len(idTestx)):
    testFeatures[i,0] = get_distance(test["query"][i],test["product_title"][i])
    testFeatures[i,1] = get_distance(test["query"][i],test["product_description"][i])
    testFeatures[i,2] = len(test["query"][i].split(" "))
    testFeatures[i,3] = score_title(test["query"][i],test["product_title"][i])

# drop ID, median_relevance and relevance_variance
test = test.drop('id', axis=1)
train = train.drop(['id','median_relevance', 'relevance_variance'], axis=1)


## PORTER STEMMING ##

stemmer = PorterStemmer()

for i in range(len(y)):
    p=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")])
    p=re.sub("[^a-zA-Z0-9]"," ", p)
    p= (" ").join([stemmer.stem(z) for z in p.split(" ")])
    s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")])
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    traindata2.append(s)
    traindata1.append(p)

for i in range(len(idTestx)):
    p=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text(" ").split(" ")])
    p=re.sub("[^a-zA-Z0-9]"," ", p)
    p= (" ").join([stemmer.stem(z) for z in p.split(" ")])
    traindata1.append(p)
    s=(" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text(" ").split(" ")])
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    traindata2.append(s)

# Initialize tf-idf vectorization function

tfv = TfidfVectorizer(min_df=1,max_features=None,strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,2), use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words = 'english')

# Creating a Sparse Matrix for only query terms 
tfv.fit(traindata1)
X1 = tfv.transform(traindata1)

# Creating a Sparse Matrix for only product_title terms
tfv.fit(traindata2)
X2 = tfv.transform(traindata2)

# Incorporate Sparse Reduction for product_title
X2 = X2[:,numpy.squeeze(numpy.array(X2.sum(axis=0) > 3.5))]

print X2.shape
# Splitting Corpus into train and test again

X1Test  = X1[len(y)+1:]
X1      = X1[:len(y)]
X2Test  = X2[len(y)+1:]
X2      = X2[:len(y)]

# Passing the vectorized matrices to SVD
svd = TruncatedSVD(n_components = 800)
svd.fit(X1)
X1 = svd.transform(X1)
#X1Test  = svd.transform(X1Test)
svd = TruncatedSVD(n_components = 1200)
svd.fit(X2)
X2 = svd.transform(X2)
#X2Test = svd.transform(X2Test)

# Initialize Model Variables #

clf = pipeline.Pipeline([('scl', StandardScaler()),('svm', SVC(C=10,gamma=0.0002))])

#Horizontally stacking the two matrices
X = hstack((X1,X2,trainFeatures))
#X_test = hstack((X1Test,X2Test,testFeatures))

stemPred = cross_val_predict(clf,X,y,cv=2,n_jobs=-1)

print "Kappa Score for Training Data\nStemming\nScore=%f" %(quadratic_weighted_kappa(y, stemPred))
