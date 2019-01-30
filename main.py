from data_loader import DataLoader
from data_loader import Type

import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
import nltk

from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer

dataload = DataLoader(Type.ARTICLE_SET)
dataload.fit("./project/project/train.csv","./project/project/test.csv")

X_train,y_train,X_test,y_test = dataload.get_data()
print(len(y_train))
n=4300
X_test=X_train[n:]
X_train=X_train[0:n]

y_test=y_train[n:]
y_train=y_train[0:n]

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



def rem_num(X):
    for x in X:
        x = re.sub('[0-9]*', '#num', x)
    return X

#from sklearn_helpers import train_test_and_evaluate

tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)
count_vect = CountVectorizer(tokenizer=tokenizer.tokenize)     
classifier = SGDClassifier(loss='squared_hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=10, tol=None)
classifier = LogisticRegression()


text_clf = Pipeline([
     ('rem_num', pipelinize(rem_num)),
     ('vect', count_vect),
     ('tfidf', TfidfTransformer()),
     ('clf', classifier),])

text_clf.fit(X_train, y_train) 
#confusion_matrix = train_test_and_evaluate(text_clf, X_train, y_train, X_test, y_test)
predicted = text_clf.predict(X_test)
print(np.mean(predicted == y_test))
