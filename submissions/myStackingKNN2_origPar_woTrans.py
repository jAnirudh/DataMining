# Stemming with Porter Stemming 
# with two fold cross validation 
# Stacked with
# SVD with 400 components, Standard Scaling and 
# KNN with 2 neighbors and two fold cross validation

from pandas import read_csv, DataFrame
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier,DistanceMetric
from sklearn.feature_extraction.text import TfidfVectorizer
from ml_metrics import quadratic_weighted_kappa
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline, grid_search, metrics
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
from math import floor
import re

traindata = []
testdata  = []

# Read the given Data Sets

train = read_csv('../train.csv').fillna("")
test = read_csv('../test.csv').fillna("")

y = train.median_relevance.values     # Relevance Ratings of Training Data

## Pre-Processing of the Data ##

idTestx = test.id.values.astype(int)    # isolating the IDs of the Test Data

# we dont need ID columns
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# create labels. drop useless columns
train = train.drop(['median_relevance', 'relevance_variance'], axis=1)


## PORTER STEMMING ##

stemmer = PorterStemmer()

for i in range(len(y)):
    s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    traindata.append(s)
for i in range(len(idTestx)):
    s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text(" ")
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    testdata.append(s)

# Initialize tf-idf vectorization function

tfv = TfidfVectorizer(min_df=5,max_df=500, max_features=None,strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words = 'english')

tfv.fit(traindata)
X = tfv.transform(traindata)
X_test = tfv.transform(testdata)

# Initialize model Variables#

tSVD = TruncatedSVD()         # Initialize SVD
scl = StandardScaler()        # Initialize the standard scaler 
svm = SVC()                   # Initialize SVC
knn = KNeighborsClassifier()  # Initialize the KNN

# create sklearn pipeline
clf = pipeline.Pipeline([('tSVD', tSVD),('scl', scl),('svm', svm)])

# Parameter grid for Stemming
param_grid = {'svm__C':[10],'tSVD__n_components':[200]}

# scorer
kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa,greater_is_better = True)

# Stemming model
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, scoring = kappa_scorer, refit = True, cv = 2, n_jobs = -1)

# Fit model
model.fit(X,y)
model.best_estimator_.fit(X,y)
stemPred = model.best_estimator_.predict(X_test)

## KNN PREDICTOR ##

# do some lambda magic on text columns

traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']), axis=1))
testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

# Fit TFIDF

tfv = TfidfVectorizer(min_df=3, max_features=None,strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words = 'english')

tfv.fit(traindata)
#X = tfv.transform(traindata)
#X_test = tfv.transform(testdata)

clf = pipeline.Pipeline([('tSVD',tSVD),('scl',scl),('knn',knn)])
param_grid = {'knn__n_neighbors':[2],'knn__metric':[DistanceMetric.get_metric('minkowski')],'tSVD__n_components':[400]}

model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, scoring = kappa_scorer, refit = True, cv = 2, n_jobs = -1)

# Fit Model

model.fit(X, y)
#model.best_estimator_.fit(X,y)
#trainPred = model.best_estimator_.predict(X_test)
trainPred = model.predict(X_test)

# Averaging predicted relevance values

finalPred = [int(floor((int(stemPred[i])+trainPred[i])*0.5)) for i in range(len(stemPred))]

print "Kappa Score for Training Data\nStemming+KNN\nScore=%f" %(quadratic_weighted_kappa(y, finalPred))

    # Create submission file
submission = DataFrame({"id": idTestx, "prediction": finalPred})
submission.to_csv("stacking_porterStemming_knn_OrigParameters.csv", index=False)
