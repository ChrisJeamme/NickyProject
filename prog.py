reduce_item_number = 900 #1 = fullset ; 10 = 1/10 of the set

################## IMPORTS ##################

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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter

# from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
# from sklearn import decomposition, ensemble
# import xgboost, textblob
# from keras.preprocessing import text, sequence
# from keras import layers, models, optimizers

################## FUNCTIONS ##################

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

################## CLASSES ##################

class Type(Enum):
    REAL_CSV_SETS = 1
    TEST_ON_TRAINING_SET = 2
    ARTICLE_SET = 3

class ArticlePredictorBase:
    def __init__(self,test_type,path_to_csv_train,path_to_csv_test):
        if Type.ARTICLE_SET == test_type:
            
            #Import of the test set
            try:
                test = pd.read_csv(path_to_csv_test,delimiter=',')
                filenames = test.iloc[:,1]
                self.test_x = []
                for fn in filenames:
                    self.test_x.append(get_xml_text(fn))
                
                self.test_y = test.iloc[:,2]
            except:
                print("Error while importing testing set csv file") 
                self.type = None
            
            #Import of the train set
            try:
                train = pd.read_csv(path_to_csv_train,delimiter=',')
                filenames = train.iloc[:,1]
                self.train_x = []
                for fn in filenames:
                    self.train_x.append(get_xml_text(fn))
                self.train_y = train.iloc[:,2]
            except:
                print("Error while importing training set csv file") 
                self.type = None
                
            self.type = test_type
            
        elif Type.REAL_CSV_SETS == test_type:
            
            #Import of the test set
            try:
                test = pd.read_csv(path_to_csv_test,delimiter=',')
                self.test_x = test.iloc[:,1]
                self.test_y = test.iloc[:,2]
            except:
                print("Error while importing testing set csv file") 
                self.type = None
            
            #Import of the train set
            try:
                train = pd.read_csv(path_to_csv_train,delimiter=',')
                self.train_x = train.iloc[:,1]
                self.train_y = train.iloc[:,2]
            except:
                print("Error while importing training set csv file") 
                self.type = None
                
            self.type = test_type
                
        elif Type.TEST_ON_TRAINING_SET == test_type:
            
            #Import of the train/testing set
            try:
                data = pd.read_csv(path_to_csv_train,delimiter=',')
            except:
                print("Error while importing the csv file") 
                self.type = None
                
            self.type = test_type
                
            X = data.iloc[:,1]
            y = data.iloc[:,2]
            
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, y)
        else:
            self.type = None
            print('Error type')

    def preprocess(self, set_x, set_y):
        the_set = pd.DataFrame()
        the_set['file_name'] = list(set_x)
        the_set['deleted_stop_words'] = list(None for _ in range(the_set['file_name'].size))
        
        rover=0
        for i in range(rover,the_set['file_name'].size//reduce_item_number):
            # Path to the file of this element
            file_path = 'project/project/data/'+str(the_set['file_name'][i])
            # Content of the BODY element in the file
            itemlist = minidom.parse(file_path).getElementsByTagName('BODY')
            
            if len(itemlist) != 1:
                print('Error: XML file invalid')

            if itemlist[0].childNodes == []: #If the text is empty in the article
                continue
                
            text = itemlist[0].childNodes[0].nodeValue
            
            #Preprocess part
            
            text = text.lower() #Lowercase
            text = re.sub(r'\d+', '', text) #Deleting numbers
            text = text.translate(str.maketrans('','',string.punctuation)) #Deleting ponctuation

            #Tokenization    
            tokens = word_tokenize(text)
            before_stop = len(tokens)
            tokens = [i for i in tokens if not i in ENGLISH_STOP_WORDS]
            
            #Feature adding
            deleted_stop_words = before_stop/len(tokens)
            the_set['deleted_stop_words'][i] = before_stop/len(tokens)
            
            tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
            #tfidf_vect.fit(text)
            #xtrain_tfidf =  tfidf_vect.transform(????)
            #print(xtrain_tfidf)
            
            # Exemple pour récupérer les termes les plus fréquents
            # print(most_common_terms(tokens))
            
            
            #TODO preprocess (compute des trucs, ajouter des champs dans le set comme plus haut (deleted_stop_words))
       
        #print(the_set)
        return the_set, list(set_y)  #TODO Renvoyer X avec tous les nouveaux fields
            
    def test(self):
        print("(Testing Launched)")
        if self.type is None:
            print("Initialisation error ")
            raise

        test_x, test_y = self.preprocess(self.test_x, self.test_y)
        #TODO testing
        
    def train(self):
        print("(Training Launched)")
        if self.type is None:
            print("Initialisation error ")
            raise
            
        print(str(self.train_x.size)+" elements in the training set")
        train_x, train_y = self.preprocess(self.train_x, self.train_y)        
        print(train_x)   
        print(train_y)
                            
        #TODO training

################## PROGRAM ##################

#predictor = ArticlePredictorBase(Type.REAL_CSV_SETS,"project/project/train.csv","project/project/test.csv")
predictor = ArticlePredictorBase(Type.TEST_ON_TRAINING_SET,"project/project/train.csv",None)
predictor.train()

#predictor = ArticlePredictorBase(Type.ARTICLE_SET,"project/project/train.csv",None)
#predictor
