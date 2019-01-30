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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer

dataload = DataLoader(Type.ARTICLE_SET)
dataload.fit("./project/project/train.csv","./project/project/test.csv")

X,y,X_test,y_test = dataload.get_data()
# print(len(y_train))
n=4300
X_train=X[0:n]
y_train=y[0:n]

X_test=X[n:]
y_test=y[n:]

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



#from sklearn_helpers import train_test_and_evaluate

tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)
count_vect = CountVectorizer(tokenizer=tokenizer.tokenize,stop_words='english'
                             ,ngram_range=(1, 2)
                             ,min_df=1
                             #,max_features = 20
                             )
classifier = SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None)
#classifier = MultinomialNB()

text_clf = Pipeline([
     ('sub', pipelinize(sub)),
     ('vect', count_vect),
     ('tfidf', TfidfTransformer()),
     ('clf', classifier),])

# print(y_test)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import metrics

text_clf.fit(X_train, y_train) 

print(count_vect.get_feature_names())
#confusion_matrix = train_test_and_evaluate(text_clf, X_train, y_train, X_test, y_test)
predicted = text_clf.predict(X_test)
print(metrics.confusion_matrix(predicted,y_test))
print(metrics.classification_report(y_test, predicted))
print(np.mean(predicted == y_test))
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
