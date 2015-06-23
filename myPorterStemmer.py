from pandas import read_csv, DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
from ml_metrics import quadratic_weighted_kappa
from sklearn.cross_validation import cross_val_predict

#load data
train = read_csv("../train.csv").fillna("")
test  = read_csv("../test.csv").fillna("")
y = train.median_relevance.values

## Pre-Processing of the Data ##

idTestx = test.id.values.astype(int)# isolating the IDs of the Test Data

# drop ID, median_relevance and relevance_variance
test = test.drop('id', axis=1)
train = train.drop(['id','median_relevance', 'relevance_variance'], axis=1)


## PORTER STEMMING ##

# Create DataFrames for stemmed data
stemmed_train = DataFrame(index=range(len(train["query"])),columns = list(train.columns.values))
stemmed_test  = DataFrame(index=range(len(train["query"])),columns = list(test.columns.values))
stemmer = PorterStemmer()

for j in list(train.columns.values):
    for i in range(len(y)):
        s=(" ").join([z for z in BeautifulSoup(train[j][i]).get_text(" ").split(" ")])
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        stemmed_train[j][i] = s
    for i in range(len(idTestx)):
        s=(" ").join([z for z in BeautifulSoup(test[j][i]).get_text(" ").split(" ")])
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        stemmed_test[j][i] = s

# do some lambda magic on text columns

#traindata = list(stemmed_train.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'], x['product_description']),axis=1))
#testdata = list(stemmed_test.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'], x['product_description']),axis=1))
traindata = list(stemmed_train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
testdata = list(stemmed_test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))



# Initialize tf-idf vectorization function

tfv = TfidfVectorizer(min_df=3,  max_features=None,strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words = 'english')
 
tfv.fit(traindata)
X = tfv.transform(traindata)

# Initialize Model Variables #

tSVD = TruncatedSVD(n_components=400)
scl  = StandardScaler()
svm  = SVC(C=10) 

#create sklearn pipeline

clf = pipeline.Pipeline([('tSVD', tSVD),('scl', scl),('svm', svm)])

stemPred = cross_val_predict(clf,X,y,cv=2,n_jobs=-1)

print "Kappa Score for Training Data\nStemming\nScore=%f" %(quadratic_weighted_kappa(y, stemPred))
