from data_loader import DataLoader
from data_loader import Type

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
import nltk
nltk.download('wordnet')

# import en_core_web_sm
# nlp = en_core_web_sm.load()

from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network  import MLPClassifier
import xgboost as xgb

from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer

dataload = DataLoader(Type.ARTICLE_SET)
dataload.fit("./project/project/train.csv","./project/project/test.csv")

X,y,X_test,y_test = dataload.get_data()

X_train, X_test, y_train, y_test = train_test_split(X, y)

### Cross validation
nb_splits = 5
kf = KFold(n_splits=nb_splits, shuffle=True)


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

x_count = count_vect.transform(X_test)
x_tf = tf_transformer.transform(x_count)

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
    x = re.sub(r'(\#num)+', '#num', x)
    x = re.sub(r'[Nn]ew[ -][Yy]ork','NY',x)
    return x


# tokenize the doc and lemmatize its tokens
def my_tokenizer(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tokens = list(map(lem.lemmatize_tagged_token, tagged))
    return(tokens)

count_vect = CountVectorizer(tokenizer=my_tokenizer
                             ,stop_words='english'
                             ,ngram_range=(1, 2)
                             ,min_df=1
                             #,max_features = 20
                             )

# classifier = SGDClassifier(loss='hinge', penalty='l2',
#                            alpha=1e-3, random_state=42,
#                            max_iter=5, tol=None)
# classifier = MultinomialNB()
# classifier = svm.SVC(kernel="linear")
# classifier = RandomForestClassifier(criterion="entropy")
# classifier = xgb.XGBClassifier(booster="gbtree")
classifier = MLPClassifier(activation='logistic', alpha=1e-02, batch_size='auto',
                                beta_1=0.9, beta_2=0.999, early_stopping=False,
                                epsilon=1e-02, hidden_layer_sizes=(2, 2),
                                learning_rate='constant', learning_rate_init=0.001,
                                max_iter=200, momentum=0.9,
                                nesterovs_momentum=True, power_t=0.5, random_state=1,
                                shuffle=True, solver='lbfgs', tol=0.0001,
                                validation_fraction=0.1, verbose=False, warm_start=False)

text_clf = Pipeline([
     ('sub', pipelinize(sub)),
     ('vect', count_vect),
     ('tfidf', TfidfTransformer()),
     ('clf', classifier),])

# print(y_test)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import metrics

# text_clf.fit(X_train, y_train) 

sum41 = 0
for learn,test in kf.split(X):
    text_clf.fit(X[learn],y[learn])
    predicted = text_clf.predict(X[test])
    print(metrics.confusion_matrix(predicted,y[test]))
    print(metrics.classification_report(y[test], predicted))
    sum41 += np.mean(predicted == y[test]) 
    print(np.mean(predicted == y[test]))

print('Average precision:' + str(sum41/nb_splits))
# print(count_vect.get_feature_names())
#confusion_matrix = train_test_and_evaluate(text_clf, X_train, y_train, X_test, y_test)

# from sklearn.cross_validation import cross_val_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler

# results = []

# for n in range(1, 50, 2):
#    pipe = make_pipeline(StandardScaler(),
#    KNeighborsClassifier(n_neighbors=n))
#    c_val = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
#    results.append([n, c_val])

# print(results)
