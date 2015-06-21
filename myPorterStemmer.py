from pandas import read_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
from ml_metrics import quadratic_weighted_kappa
from sklearn.cross_validation import cross_val_predict

# array declarations
traindata = []

if __name__ == '__main__':
    
    #load data
    train = read_csv("../train.csv").fillna("")

    y = train.median_relevance.values    
    
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

    # Initialize Model Variables #

    tSVD = TruncatedSVD(n_components=400)
    scl  = StandardScaler()
    svm  = SVC(C=10) 

    #create sklearn pipeline

    clf = pipeline.Pipeline([('tSVD', tSVD),('scl', scl),('svm', svm)])
    
    stemPred = cross_val_predict(clf,X,y,cv=2,n_jobs=-1)
    
    print "Kappa Score for Training Data\nStemming\nScore=%f" %(quadratic_weighted_kappa(y, stemPred))
