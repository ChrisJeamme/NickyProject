from data_loader import DataLoader
from data_loader import Type

import csv
import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm
import lem
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network  import MLPClassifier
import xgboost as xgb
import datetime
from numpy import genfromtxt

from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import ShuffleSplit

#------------------#
### Type of test ###
#------------------#
 
# type =>
# Type.REAL_CSV_SETS = Predict the test.csv file
# Type.TEST_ON_TRAINING_SET = Test a classifier with a kfold cross validator to get an average accuracy and other informations

##########################
type = Type.REAL_CSV_SETS
##########################

if(type == Type.REAL_CSV_SETS):
    dataload = DataLoader(Type.REAL_CSV_SETS)
else:
    dataload = DataLoader(Type.TEST_ON_TRAINING_SET)

# ------------------#
### Load the data ###
# ------------------#

dataload.fit("./project/project/train.csv","./project/project/test.csv")
X, y, X_train, y_train, X_test, y_test = dataload.get_data()

# --------------#
### Functions ###
# --------------#

def pipelinize(function, active=True):
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            return [function(i) for i in list_or_series]
        else: # if it's not active, just pass it right back
            return list_or_series
    return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active':active})

def sub(x):
    
    x = x.replace('\n',' ') 
    x = x.replace('\'s','')       
    x = re.sub('[^A-Za-z0-9]+', ' ', x)
    
    x = re.sub('\d+,\d+|\d+.\d+|\d+', 'number', x)
    x = x.replace('yr ago','year ago')
    
    x = re.sub(r'(number|mln|billion) (number|mln|billion)', ' number', x)
    x = re.sub(r'[Nn]ew[ -][Yy]ork','NY',x)
    return x

# tokenize the doc and lemmatize its tokens
def my_tokenizer(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tokens = list(map(lem.lemmatize_tagged_token, tagged))
    return(tokens) 

def predict_test_csv(X_train, y_train, X_test):
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    return predicted

# --------------#
### Cross Val ###
# --------------#

def cross_val(clf,X,y):
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    pred = cross_val_predict(text_clf, X_train, y_train, cv=5)
    print(metrics.confusion_matrix(y_train, pred))
    print(np.mean(y_train == pred))
    
# def classifier_test_accuracy(text_clf, X, y):
#     sum41 = 0
#     for learn,test in kf.split(X):
#         text_clf.fit(X[learn],y[learn])
#         predicted = text_clf.predict(X[test])
#         print(metrics.confusion_matrix(predicted,y[test]))
#         print(metrics.classification_report(y[test], predicted))
#         sum41 += np.mean(predicted == y[test]) 
#         print(np.mean(predicted == y[test]))
#     print('Average accuracy:' + str(sum41/nb_splits))

def csv_write(y_test):
    test_csv = pd.read_csv('project/project/test.csv','r', delimiter=",")

    for i, line in enumerate(test_csv.values):
        test_csv.at[i,'earnings'] = y_test[i]
        
    test_csv['earnings'] =test_csv['earnings'].astype(int)

    filename = 'project/project/test_'+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+'.csv'
    test_csv.to_csv(path_or_buf=filename, index=False)
    print("Filled test.csv was created : " + 'project/project/test_'+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+'.csv')


# ----------------------#
### Classifier choice ###
# ----------------------#

classifier1 = SGDClassifier(max_iter=5, tol=None)
classifier2 = SGDClassifier(alpha=0.0001, average=False, class_weight=None,
                            epsilon=0.1, eta0=0.0, fit_intercept=True,
                            l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=10,
                            n_iter=None, penalty='l2',
                            power_t=0.5, random_state=None, shuffle=False, tol=0.0001, verbose=0, warm_start=False)
classifier3 = MultinomialNB()
classifier4 = svm.SVC(kernel="linear")
classifier5 = RandomForestClassifier(criterion="entropy")
classifier6 = xgb.XGBClassifier(booster="gbtree")
classifier7 = MLPClassifier(activation='logistic', alpha=0.01, batch_size='auto',
                                beta_1=0.9, beta_2=0.999, early_stopping=False,
                                epsilon=1e-01, hidden_layer_sizes=(2,2),
                                learning_rate='constant', learning_rate_init=0.01,
                                max_iter=200, momentum=0.9,
                                nesterovs_momentum=True, power_t=0.5, random_state=1,
                                shuffle=True, solver='lbfgs', tol=0.0001,
                                validation_fraction=0.1, verbose=False, warm_start=False)

##################################
classifier_choosen = classifier2
##################################

# ------------------#
### Preprocessing ###
# ------------------#
tft = TfidfTransformer(sublinear_tf=True)
count_vect = CountVectorizer(tokenizer=my_tokenizer,stop_words='english',analyzer='word'
                             ,ngram_range=(1, 3)
                             ,min_df=3,max_df=0.7
                             ,lowercase = True
                             #,max_features = 100
                             )
# ------------------------#
### Pipeline definition ###
# ------------------------#

text_clf = Pipeline([
     ('sub', pipelinize(sub)),
     ('vect', count_vect),
     ('tfidf', tft),
    #  ('svd', TruncatedSVD(n_components=50)),
     ('clf', classifier_choosen),])

#----------------#
### Processing ###
# ---------------#

if(type == Type.REAL_CSV_SETS):
    print("Predicting test.csv ...")
    y_test = predict_test_csv(X_train, y_train, X_test)
    csv_write(y_test)
else:
    print("Accuracy testing ...")
    cross_val(text_clf,X_train,y_train)