from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
from nltk.stem.porter import PorterStemmer
import re
from bs4 import BeautifulSoup
from ml_metrics import quadratic_weighted_kappa
from sklearn.ensemble import RandomForestClassifier

# array declarations
s_data = []
s_labels = []
t_data = []
t_labels = []

if __name__ == '__main__':

    # Load the training file
    train = pd.read_csv('../train.csv')
    test = pd.read_csv('../test.csv')
    
    # we dont need ID columns
    idx = test.id.values.astype(int)
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)
    
    # create labels. drop useless columns
    y = train.median_relevance.values
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)
    
    # do some lambda magic on text columns
    traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    
    # the infamous tfidf vectorizer (Do you remember this one?)
    tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
    
    # Fit TFIDF
    tfv.fit(traindata)
    X =  tfv.transform(traindata) 
    X_test = tfv.transform(testdata)
    
    # Initialize SVD
    svd = TruncatedSVD(n_components = 400)
    
    # Initialize the standard scaler 
    scl = StandardScaler()
    
    # We will use RF here..
    rf = RandomForestClassifier(n_estimators = 100)
    
    # Create the pipeline 
    clf = pipeline.Pipeline([('svd', svd),
    						 ('scl', scl),
                    	     ('rf', rf)])
    
    # Create a parameter grid to search for best parameters for everything in the pipeline
#    param_grid = {'svd__n_components' : [400],
                  #'svm__C': [10]}
    
    # Kappa Scorer 
#    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)
    
    # Initialize Grid Search Model
#    model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

    
    
    # Fit Grid Search Model
   # model.fit(X, y)
   # print("Best score: %0.3f" % model.best_score_)
   # print("Best parameters set:")
   # best_parameters = model.best_estimator_.get_params()
   # for param_name in sorted(param_grid.keys()):
   # 	print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    # Get best model
    #best_model = model.best_estimator_
    
    # Fit model with best parameters optimized for quadratic_weighted_kappa
    #best_model.fit(X,y)
    clf.fit(X,y)
    #preds = best_model.predict(X)
    preds = clf.predict(X_test)
    #load data
    train = pd.read_csv("../train.csv").fillna("")
    test  = pd.read_csv("../test.csv").fillna("")
    
    #remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)
    stemmer = PorterStemmer()
    ## Stemming functionality
    class stemmerUtility(object):
        """Stemming functionality"""
        @staticmethod
        def stemPorter(review_text):
            porter = PorterStemmer()
            preprocessed_docs = []
            for doc in review_text:
                final_doc = []
                for word in doc:
                    final_doc.append(porter.stem(word))
                    #final_doc.append(wordnet.lemmatize(word)) #note that lemmatize() can also takes part of speech as an argument!
                preprocessed_docs.append(final_doc)
            return preprocessed_docs
    
    
    for i in range(len(train.id)):
        s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s_data.append(s)
        s_labels.append(str(train["median_relevance"][i]))
    for i in range(len(test.id)):
        s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        t_data.append(s)
    #create sklearn pipeline, fit all, and predit test data
    clf = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
    ('svd', TruncatedSVD(n_components=200, algorithm='randomized', random_state=None, tol=0.0)), 
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
    ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
    clf.fit(s_data, s_labels)
    t_labels = clf.predict(t_data)
    
    import math
    p3 = []
    for i in range(len(preds)):
        x = (int(t_labels[i]) + preds[i])/2
        x = math.floor(x)
        p3.append(int(x))
        
        
    
    # p3 = (t_labels + preds)/2
    # p3 = p3.apply(lambda x:math.floor(x))
    # p3 = p3.apply(lambda x:int(x))
    
    #print "Kappa Score for Training Data\nStemming+KNN\nScore=%f" %(quadratic_weighted_kappa(y, p3)) 

    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": p3})
    submission.to_csv("stacking_RandomForest_stemming.csv", index=False)
