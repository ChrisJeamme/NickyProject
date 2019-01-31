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

# ---------------------#
### Cross-validation ###
# ---------------------#

nb_splits = 5
kf = KFold(n_splits=nb_splits, shuffle=True)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

x_count = count_vect.transform(X_test)
x_tf = tf_transformer.transform(x_count)

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
    x = re.sub('\d+,\d+|\d+.\d+|\d+', '#num ', x)
    x = x.replace('/','')
    x = x.replace('+','')
    x = x.replace('.','')
    x = x.replace(',','')
    x = x.replace('``','')
    x = x.replace('"','')
    x = x.replace('<','')
    x = x.replace('>','')
    x = x.replace('(','')
    x = x.replace(')','')
    x = x.replace('yr-ago','year ago')
    x = re.sub(r'(\#num)+', '#num', x)
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

def classifier_test_accuracy(text_clf, X, y):
    sum41 = 0
    for learn,test in kf.split(X):
        text_clf.fit(X[learn],y[learn])
        predicted = text_clf.predict(X[test])
        print(metrics.confusion_matrix(predicted,y[test]))
        print(metrics.classification_report(y[test], predicted))
        sum41 += np.mean(predicted == y[test]) 
        print(np.mean(predicted == y[test]))
    print('Average accuracy:' + str(sum41/nb_splits))

def csv_write(y_test):
    spamwriter = csv.writer(open('test_1.csv', 'w', newline=''), delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for value in y_test:
        spamwriter.writerow(str(value))

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
classifier_choosen = classifier1
##################################

# ------------------#
### Preprocessing ###
# ------------------#

count_vect = CountVectorizer(tokenizer=my_tokenizer
                             ,stop_words='english'
                             ,ngram_range=(1, 3)
                             ,min_df=1
                             ,lowercase=True
                             #,max_features = 20
                             )
# ------------------------#
### Pipeline definition ###
# ------------------------#

text_clf = Pipeline([
     ('sub', pipelinize(sub)),
     ('vect', count_vect),
     ('tfidf', TfidfTransformer()),
    #  ('svd', TruncatedSVD(n_components=50)),
     ('clf', classifier_choosen),])

#----------------#
### Processing ###
# ---------------#

if(type == Type.REAL_CSV_SETS):
    y_test = predict_test_csv(X_train, y_train, X_test)
    csv_write(y_test)
else:
    classifier_test_accuracy(text_clf, X, y)