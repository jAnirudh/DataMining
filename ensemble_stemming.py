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
from math import floor

# create arrays

traindata1 = []
traindata2 = []
testdata1  = []
testdata2  = []

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
    testdata1.append(p)
    s=(" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text(" ").split(" ")])
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    testdata2.append(s)

# Initialize tf-idf vectorization function

tfv = TfidfVectorizer(min_df=5,max_df=500,max_features=None,strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,2), use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words = 'english')

# Creating a Sparse Matrix for only query terms 
tfv.fit(traindata1)
X1 = tfv.transform(traindata1)
#X1_test = tfv.transform(testdata1)

# Creating a Sparse Matrix for only product_title terms
tfv.fit(traindata2)
X2 = tfv.transform(traindata2)
#X2_test = tfv.transform(testdata2)

# Passing the vectorized matrices to SVD
svd = TruncatedSVD(n_components = 400)
svd.fit(X1)
X1 = svd.transform(X1)
#X1_test = svd.transform(X1_test)

svd = TruncatedSVD(n_components = 800)
svd.fit(X2)
X2 = svd.transform(X2)
#X2_test = svd.transform(X2_test)

# Initialize Model Variables #

clf = pipeline.Pipeline([('scl', StandardScaler()),('svm', SVC(C=10,gamma=0.0002))])

#Horizontally stacking the two matrices
X = hstack((X1,X2,trainFeatures))
#X_test = hstack((X1_test,X2_test,testFeatures))

stemPred = cross_val_predict(clf,X,y,cv=2,n_jobs=-1)

print "Kappa Score for Training Data\nStemming\nScore=%f" %(quadratic_weighted_kappa(y, stemPred))

## SVM Predictor ##

traindata1 = list(train.apply(lambda x:'%s' % (x['query']), axis=1))
testdata1  = list(test.apply(lambda x:'%s' % (x['query']),axis=1))
traindata2 = list(train.apply(lambda x:'%s' % (x['product_title']), axis=1))
testdata2  = list(test.apply(lambda x:'%s' % (x['product_title']),axis=1))

# Creating a Sparse Matrix for only query terms 
tfv.fit(traindata1)
X1 = tfv.transform(traindata1)
#X1_test = tfv.transform(testdata1)

# Creating a Sparse Matrix for only product_title terms
tfv.fit(traindata2)
X2 = tfv.transform(traindata2)
#X2_test = tfv.transform(testdata2)

# Passing the vectorized matrices to SVD
svd = TruncatedSVD(n_components = 400)
svd.fit(X1)
X1 = svd.transform(X1)
#X1_test = svd.transform(X1_test)

svd = TruncatedSVD(n_components = 800)
svd.fit(X2)
X2 = svd.transform(X2)
#X2_test = svd.transform(X2_test)

#Horizontally stacking the two matrices
X = hstack((X1,X2,trainFeatures))
#X_test = hstack((X1_test,X2_test,testFeatures))

Pred = cross_val_predict(clf,X,y,cv=2,n_jobs=-1)
print "Kappa Score for Training Data\nWithout Stemming\nScore=%f" %(quadratic_weighted_kappa(y, Pred))

# Averaging predicted relevance values

finalPred = [int(floor((int(stemPred[i])+Pred[i])*0.5)) for i in range(len(stemPred))]
print "Kappa Score for Training Data\nCombined\nScore=%f" %(quadratic_weighted_kappa(y, finalPred))

