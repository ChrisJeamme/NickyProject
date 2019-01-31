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

##########################
type = Type.TEST_ON_TRAINING_SET
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

count_vect = CountVectorizer(tokenizer=my_tokenizer,stop_words='english',analyzer='word'
                             ,ngram_range=(1, 3)
                             ,min_df=3,max_df=0.7
                             #,max_features = 100
                             )
tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False,
                     norm=None, analyzer='word')
tf = TfidfVectorizer(strip_accents='unicode',stop_words='english',
                     tokenizer=my_tokenizer, ngram_range=(1, 3),
                     max_df=0.7, min_df=3, sublinear_tf=True)
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

    
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import metrics

from sklearn.model_selection import cross_val_score,cross_val_predict

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
pred = cross_val_predict(text_clf, X_train, y_train, cv=5)
conf_matrix = metrics.confusion_matrix(y_train, pred)
print(conf_matrix)
print(np.mean(y_train == pred))

#Apply this vectorizer to text to get a sparse matrix of counts
X = []
for x in X_train:
    x = sub(x)
    X.append(x)

count_vect = CountVectorizer(tokenizer=my_tokenizer,stop_words='english',analyzer='word'
                             ,ngram_range=(1, 3)
                             ,min_df=3,max_df=0.7
                             #,max_features = 100
                             )
count_matrix = count_vect.fit_transform(X)
#Get the names of the features
features = count_vect.get_feature_names()
features = np.asanyarray(features)
sum_ = [sum(x) for x in zip(*count_matrix.toarray())]
#Create a series from the sparse matrix
d = pd.Series(sum_, 
              index = features).sort_values(ascending=False)

ax = d[:20].plot(kind='bar', figsize=(10,6), width=.8, fontsize=14, rot=45,
            title='Article Word Counts')
ax.title.set_size(18)



import itertools
import numpy as np
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Plot non-normalized confusion matrix
    
class_names = ['non-earning','earning']
plt.figure()
plot_confusion_matrix(conf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()