from pandas import read_csv, DataFrame
from bs4 import BeautifulSoup
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import hstack
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline, grid_search, metrics
from ml_metrics import quadratic_weighted_kappa
from sklearn.cross_validation import cross_val_predict
from math import floor

# create arrays

traindata1 = []
traindata2 = []
testdata1  = []
testdata2  = []

#load data
train = read_csv("train.csv").fillna("")
test  = read_csv("test.csv").fillna("")
y = train.median_relevance.values

## Pre-Processing of the Data ##

idTestx = test.id.values.astype(int)# isolating the IDs of the Test Data

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
X1_test = tfv.transform(testdata1)

# Creating a Sparse Matrix for only product_title terms
tfv.fit(traindata2)
X2 = tfv.transform(traindata2)
X2_test = tfv.transform(testdata2)

# Passing the vectorized matrices to SVD
svd = TruncatedSVD(n_components = 400)
svd.fit(X1)
X1 = svd.transform(X1)
X1_test = svd.transform(X1_test)

svd = TruncatedSVD(n_components = 800)
svd.fit(X2)
X2 = svd.transform(X2)
X2_test = svd.transform(X2_test)

# Reading data given by Bhavesh
addFeature_train = read_csv("anirudh.csv")
addFeature_test  = read_csv("anirudhTest.csv")

# Initialize Model Variables #

clf = pipeline.Pipeline([('scl', StandardScaler()),('svm', SVC(gamma=0.0005))])

#Horizontally stacking the two matrices
X = hstack((X1,X2,addFeature_train.as_matrix()))
X_test = hstack((X1_test,X2_test,addFeature_test.as_matrix()))

#stemPred = cross_val_predict(clf,X,y,cv=2,n_jobs=-1)
param_grid = {'svm__C':[10]}
kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa,greater_is_better = True)
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, scoring = kappa_scorer, refit = True, cv = 2, n_jobs = -1)
model.fit(X,y)
model.best_estimator_.fit(X,y)
stemPred = model.best_estimator_.predict(X_test)
#print "Kappa Score for Training Data\nStemming\nScore=%f" %(model.best_score_)

#print "Kappa Score for Training Data\nStemming\nScore=%f" %(quadratic_weighted_kappa(y, stemPred))

## SVM Predictor ##

traindata1 = list(train.apply(lambda x:'%s' % (x['query']), axis=1))
testdata1  = list(test.apply(lambda x:'%s' % (x['query']),axis=1))
traindata2 = list(train.apply(lambda x:'%s' % (x['product_title']), axis=1))
testdata2  = list(test.apply(lambda x:'%s' % (x['product_title']),axis=1))

# Creating a Sparse Matrix for only query terms 
tfv.fit(traindata1)
X1 = tfv.transform(traindata1)
X1_test = tfv.transform(testdata1)

# Creating a Sparse Matrix for only product_title terms
tfv.fit(traindata2)
X2 = tfv.transform(traindata2)
X2_test = tfv.transform(testdata2)

# Passing the vectorized matrices to SVD
svd = TruncatedSVD(n_components = 400)
svd.fit(X1)
X1 = svd.transform(X1)
X1_test = svd.transform(X1_test)

svd = TruncatedSVD(n_components = 800)
svd.fit(X2)
X2 = svd.transform(X2)
X2_test = svd.transform(X2_test)

#Horizontally stacking the two matrices
X = hstack((X1,X2,addFeature_train.as_matrix()))
X_test = hstack((X1_test,X2_test,addFeature_test.as_matrix()))

#Pred = cross_val_predict(clf,X_test,cv=2,n_jobs=-1)
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, scoring = kappa_scorer, refit = True, cv = 2, n_jobs = -1)
model.fit(X,y)
model.best_estimator_.fit(X,y)
Pred = model.best_estimator_.predict(X_test)

# Averaging predicted relevance values

finalPred = [int(floor((int(stemPred[i])+Pred[i])*0.5)) for i in range(len(stemPred))]

# Create submission file
submission = DataFrame({"id": idTestx, "prediction": finalPred})
submission.to_csv("stacking_addedFeatures.csv", index=False)
