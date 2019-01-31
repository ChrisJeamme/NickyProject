from data_loader import DataLoader
from data_loader import Type

import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
import lem
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk
nltk.download('averaged_perceptron_tagger')
import nltk
nltk.download('wordnet')

import en_core_web_sm
nlp = en_core_web_sm.load()

from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.model_selection import ShuffleSplit

from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer,HashingVectorizer,TfidfVectorizer

dataload = DataLoader(Type.ARTICLE_SET)
dataload.fit("./project/project/train.csv","./project/project/test.csv")

X_train,y_train,X_test,y_test = dataload.get_data()
# print(len(y_train))


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

count_vect = CountVectorizer(tokenizer=my_tokenizer,stop_words='english',analyzer='word'
                             ,ngram_range=(1, 3)
                             ,min_df=3,max_df=0.7
                             
                             #,max_features = 100
                             )
tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
tf = TfidfVectorizer(strip_accents='unicode', tokenizer=my_tokenizer, ngram_range=(1, 2), max_df=0.9, min_df=3, sublinear_tf=True)
tft = TfidfTransformer(sublinear_tf=True)
classifier = SGDClassifier(alpha=0.0001, average=False, class_weight=None,
       early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
       l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=10,
       n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l2',
       power_t=0.5, random_state=None, shuffle=False, tol=0.0001,
       validation_fraction=0.1, verbose=0, warm_start=False)

text_clf = Pipeline([
      ('sub', pipelinize(sub)),
      ('vect', count_vect),
      ('tfidf', tft),
      # ('viz',visualizer),
      ('clf', classifier),])

# text_clf = Pipeline([
#       ('sub', pipelinize(sub)),
#       ('tf', tf),
#       # ('viz',visualizer),
#       ('clf', classifier),])

# text_clf = Pipeline([
#      ('sub', pipelinize(sub)),
#      ('hashvect',HashingVectorizer(tokenizer=my_tokenizer,stop_words='english')),
#      ('clf', classifier),
#         ])
    
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import metrics

text_clf.fit(X_train, y_train)

#print(count_vect.get_feature_names())

# visualizer.poof()

from sklearn.model_selection import cross_val_score,cross_val_predict

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
pred = cross_val_predict(text_clf, X_train, y_train, cv=5)
print(metrics.confusion_matrix(y_train, pred))
print(np.mean(y_train == pred))

# print(metrics.confusion_matrix(predicted,y_test))
# print(metrics.classification_report(y_test, predicted))
# print(np.mean(predicted == y_test))
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


# count_vectorizer = CountVectorizer(tokenizer=my_tokenizer,stop_words='english',analyzer='word'
#                              ,ngram_range=(1, 2)
#                              ,min_df=1,max_df=0.7
                             
#                              ,max_features = 3000
#                              )
# #Apply this vectorizer to text to get a sparse matrix of counts
# count_matrix = count_vectorizer.fit_transform(X_train)
# #Get the names of the features
# features = count_vectorizer.get_feature_names()
# #Create a series from the sparse matrix
# d = pd.Series(count_matrix.toarray().flatten(), 
#               index = features).sort_values(ascending=False)

# ax = d[:10].plot(kind='bar', figsize=(10,6), width=.8, fontsize=14, rot=45,
#             title='Article Word Counts')
# ax.title.set_size(18)
