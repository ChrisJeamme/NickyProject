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
    xmldoc = minidom.parse('project/project/data/'+filename)
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
        return np.asarray(X),np.asarray(y)
    ############
    def __init__(self,test_type):
        self.type = test_type

    def fit(self,path_to_csv_train,path_to_csv_test):
        if Type.REAL_CSV_SETS == self.type:
            self.X_train, self.y_train = self.csv_convert(path_to_csv_train)
            self.X_test, self.y_test = self.csv_convert(path_to_csv_test)
            self.X = None
            self.y = None

        elif Type.TEST_ON_TRAINING_SET == self.type:
            self.X, self.y = self.csv_convert(path_to_csv_train)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        else:
            self.type = Type.NONE
            print('Error type')

    def get_data(self):
        return self.X, self.y, self.X_train,self.y_train,self.X_test,self.y_test
    
    def get_train(self):
        return self.X_train,self.y_train
    
    def get_test(self):
        return self.X_test,self.y_test