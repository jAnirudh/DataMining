from pandas import read_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from nltk.stem.porter import PorterStemmer
import re
from bs4 import BeautifulSoup
from ml_metrics import quadratic_weighted_kappa

# array declarations
traindata = []

if __name__ == '__main__':
    
    #load data
    train = read_csv("../train.csv").fillna("")

    y = train.median_relevance.values    
    #remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)
    stemmer = PorterStemmer()
    
    for i in range(len(y)):
        s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
#        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        traindata.append(s)
        #s_labels.append(str(train["median_relevance"][i]))

    tfv = TfidfVectorizer(min_df=3,  max_features=None,strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words = 'english')
 
    tfv.fit(traindata)
    X = tfv.transform(traindata)

    #create sklearn pipeline, fit all, and predit test data
    clf = pipeline.Pipeline([('svd', TruncatedSVD(n_components=400, algorithm='randomized', random_state=None, tol=0.0)),('scl', StandardScaler(copy=True, with_mean=True, with_std=True)),('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
    

    clf.fit(X,y)
    stemPred = clf.predict(X)
    
    print "Kappa Score for Training Data\nStemming\nScore=%f" %(quadratic_weighted_kappa(y, stemPred))
