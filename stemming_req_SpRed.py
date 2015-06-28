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
testdata1  = []
testdata2  = []
train_rel  = []


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
    train_rel.append(s)
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

#clf = pipeline.Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')),
#    ('svd', TruncatedSVD(n_components=200)),
#    ('scl', StandardScaler()),
#    ('svm', SVC(C=10.0))])

tfv = TfidfVectorizer(min_df=1,max_features=None,strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words = 'english')

# Creating a Sparse Matrix for only query terms 
tfv.fit(traindata1)
X1 = tfv.transform(traindata1)

# Creating a Sparse Matrix for only product_title terms
tfv.fit(traindata2)
X2 = tfv.transform(traindata2)

# Passing the vectorized matrices to SVD
svd = TruncatedSVD(n_components = 800)
svd.fit(X1)
X1 = svd.transform(X1)
svd = TruncatedSVD(n_components = 600)
svd.fit(X2)
X2 = svd.transform(X2)

# Initialize Model Variables #

clf = pipeline.Pipeline([('scl', StandardScaler()),('svm', SVC(C=10.0))])

#tSVD = TruncatedSVD(n_components=400)
#scl  = StandardScaler()

# pipeline for operating on X2
#transform2feature = pipeline.Pipeline([('tSVD', tSVD),('scl', scl)])

# transforming the sparse matrix of query and product_title to incorporate SVD and Scaling
#transform2feature.fit(X2)
#transform2feature.transform(X2)

#Horizontally stacking the two sparse matrices
X = hstack((X1[:len(y)],X2[:len(y)]))

#Initialize SVM model
#svm  = SVC(C=10) 

#create sklearn pipeline

#clf = pipeline.Pipeline([('scl', scl),('svm', svm)])

# Classify the sparse stacked sparse matrix using two fold cross validation and model specified in line 80
#stemPred = cross_val_predict(svm,X,y,cv=2,n_jobs=-1)

stemPred = cross_val_predict(clf,X,y,cv=2,n_jobs=-1)
print "Kappa Score for Training Data\nStemming\nScore=%f" %(quadratic_weighted_kappa(y, stemPred))
