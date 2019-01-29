from numpy import genfromtxt
import pandas as pd
import numpy as np
import operator, re, string, codecs, nltk
from statistics import mean
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt')
from xml.dom import minidom
from string import punctuation
from enum import Enum
try:
    maketrans = ''.maketrans
except AttributeError:
    from string import maketrans
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from collections import Counter
import sys
# Test

# from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn import decomposition, ensemble

# import xgboost, textblob
# from keras.preprocessing import text, sequence
# from keras import layers, models, optimizers

reduce_item_number = 900 #1 = fullset ; 10 = 1/10 of the set

def get_xml_text(filename):
    xmldoc = minidom.parse('project/project/data/1.xml')
    itemlist = xmldoc.getElementsByTagName('BODY')

    #Verification

    if len(itemlist) != 1:
        print('Error: XML file invalid')
        sys.exit(0)
    try :
        text = itemlist[0].childNodes[0].nodeValue
    except :
        text = ""
    return text



def most_common_terms(terms):
    terms_count_array = []
    common_terms = []
    for term in terms:
        terms_count_array += term.split(" ")
    counter = Counter(terms_count_array)
    for word in counter.most_common():
        common_terms.append(word)
    return common_terms

class Type(Enum):
    NONE = 0 
    REAL_CSV_SETS = 1
    TEST_ON_TRAINING_SET = 2
    ARTICLE_SET = 3
    
    
class DataLoader:
    def csv_convert(self,path):
        
        data = pd.read_csv(path,delimiter=',')
        filenames = data.iloc[:,1]
        X = []
        for fn in filenames:
            X.append(get_xml_text(fn))
        
        y = data.iloc[:,2]
        return np.asarray(X),y
    ############
    def __init__(self,test_type):
        self.type = test_type

    def fit(self,path_to_csv_train,path_to_csv_test):
        if Type.ARTICLE_SET == self.type:
            self.X_train,self.y_train = self.csv_convert(path_to_csv_train)
            self.X_test,self.y_test = self.csv_convert(path_to_csv_test)
            
        else:
            self.type = Type.NONE
            print('Error type')

    def get_data(self):
        return self.X_train,self.y_train,self.X_test,self.y_test
    
    def get_train(self):
        return self.X_train,self.y_train
    
    def get_test(self):
        return self.X_test,self.y_test
    
    

dataload = DataLoader(Type.ARTICLE_SET)
dataload.fit("./project/project/train.csv","./project/project/test.csv")

X_train,y_train,X_test,y_test = dataload.get_data()
print(len(y_train))
n=4300
X_test=X_train[n:]
X_train=X_train[0:n]

y_test=y_train[n:]
y_train=y_train[0:n]

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

print(X_train_counts[0])
print(X_train_counts.shape)
print(count_vect.vocabulary_.get(u'dlrs'))


from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf[0])


x_count = count_vect.transform(X_test)
x_tf = tf_transformer.transform(x_count)
print(y_test)

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='squared_hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=10, tol=None)),])

text_clf.fit(X_train, y_train) 
    
predicted = text_clf.predict(X_test)
np.mean(predicted == y_test)     
