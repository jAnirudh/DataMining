from pandas import read_csv, DataFrame
from createCorpus import Corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from ml_metrics import quadratic_weighted_kappa
from sklearn.cross_validation import cross_val_predict

#load data
train = read_csv("../train.csv").fillna("")
test  = read_csv("../test.csv").fillna("")
y = train.median_relevance.values

## Pre-Processing of the Data ##

##In [4]: train = train.drop([list(train.columns.values)[i] for i in [0,4,5]],axis=1)


idTestx = test.id.values.astype(int)# isolating the IDs of the Test Data

# drop ID, median_relevance and relevance_variance
test = test.drop('id', axis=1)
train = train.drop(['id','median_relevance', 'relevance_variance'], axis=1)


## PORTER STEMMING ##

# Create DataFrames for stemmed data
stemmed_train = StemmedCorpus(train,index=range(len(train["query"])),columns = list(train.columns.values))
stemmed_test  = StemmedCorpus(train,index=range(len(train["query"])),columns = list(test.columns.values))

# do some lambda magic on text columns

traindata1 = list(stemmed_train.apply(lambda x:'%s' % (x['query']),axis=1))
testdata1 = list(stemmed_test.apply(lambda x:'%s' % (x['query']),axis=1))
traindata2 = list(stemmed_train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
testdata2 = list(stemmed_test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))


# Initialize tf-idf vectorization function

tfv = TfidfVectorizer(min_df=3,  max_features=None,strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words = 'english')
 
tfv.fit(traindata1)
X1 = tfv.transform(traindata1)
tfv.fit(traindata2)
X2 = tfv.transform(traindata2)

X = hstack((X1,X2))

# Initialize Model Variables #

tSVD = TruncatedSVD(n_components=400)
scl  = StandardScaler()
svm  = SVC(C=10) 

#create sklearn pipeline

clf = pipeline.Pipeline([('tSVD', tSVD),('scl', scl),('svm', svm)])

stemPred = cross_val_predict(clf,X,y,cv=2,n_jobs=-1)

print "Kappa Score for Training Data\nStemming\nScore=%f" %(quadratic_weighted_kappa(y, stemPred))
