# Stemming with Porter Stemming 
# Stacked with
# SVD with 400 components, Standard Scaling and 
# KNN with 10 neighbors and two fold cross validation

from pandas import read_csv
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

traindata = []

# Read the given Data Sets

train = read_csv('../train.csv').fillna("")
#test = read_csv('../test.csv').fillna("")

y = train.median_relevance.values     # Relevance Ratings of Training Data

## Pre-Processing of the Data ##

#idTestx = test.id.values.astype(int)    # isolating the IDs of the Test Data
#idTrainx = train.id.values.astype(int)  # isolating the IDs of the Train Data

# we dont need ID columns
train = train.drop('id', axis=1)
#test = test.drop('id', axis=1)

# create labels. drop useless columns
train = train.drop(['median_relevance', 'relevance_variance'], axis=1)


## PORTER STEMMING ##

stemmer = PorterStemmer()
for i in range(len(y)):
    s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    traindata.append(s)

# Initialize tf-idf vectorization function

tfv = TfidfVectorizer(min_df=3,  max_features=None,strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words = 'english')

tfv.fit(traindata)
X = tfv.transform(traindata)

# Initialize model Variables#

tSVD = TruncatedSVD(n_components = 400)      # Initialize SVD
scl = StandardScaler()     # Initialize the standard scaler 
svm = SVC(C = 10)
knn = KNeighborsClassifier()

#create sklearn pipeline
clf = pipeline.Pipeline([('tsvd', tSVD),('scl', scl),('svm', svm)])

clf.fit(X,y)
stemPred = clf.predict(X)

## SVM Prediction without Stemming ##

# do some lambda magic on text columns

traindata = list(train.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'], x['product_description']),axis=1))
#testdata = list(test.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'], x['product_description']),axis=1))

# Fit TFIDF

tfv.fit(traindata)
X = tfv.transform(traindata) 

clf = pipeline.Pipeline([('tSVD',tSVD),('scl',scl),('knn',knn)])
param_grid = {'knn__n_neighbors':[10],'knn__metric':[DistanceMetric.get_metric('manhattan')],'tSVD__n_components':[400]}
kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa,greater_is_better = True)

model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, scoring = kappa_scorer, refit = True, cv = 2, n_jobs = -1)

# Fit Model

model.fit(X, y)
model.best_estimator_.fit(X,y)
trainPred = model.best_estimator_.predict(X)

# Averaging predicted relevance values

finalPred = [int(floor((int(stemPred[i])+trainPred[i])*0.5)) for i in range(len(stemPred))]

print "Kappa Score for Training Data\nStemming+KNN\nScore=%f" %(quadratic_weighted_kappa(y, finalPred))

