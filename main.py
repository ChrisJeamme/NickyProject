from data_loader import DataLoader
from data_loader import Type

import pandas as pd
import numpy as np

dataload = DataLoader(Type.ARTICLE_SET)
dataload.fit("./project/project/train.csv","./project/project/test.csv")

X_train,y_train,X_test,y_test = dataload.get_data()
# print(len(y_train))
n=4300
X_test=X_train[n:]
X_train=X_train[0:n]

y_test=y_train[n:]
y_train=y_train[0:n]

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

# print(X_train_counts.toarray())
# print(count_vect.get_feature_names())
# print(X_train_counts.shape)
# print(count_vect.vocabulary_.get(u'dlrs'))


from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
# print(X_train_tf[0])


x_count = count_vect.transform(X_test)
x_tf = tf_transformer.transform(x_count)
# print(y_test)

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, make_pipeline

text_clf = Pipeline([
     ('vect', CountVectorizer(strip_accents="ascii", lowercase=True, stop_words="english")),
     ('tfidf', TfidfTransformer(norm="l1", smooth_idf=True)),
     ('clf', SGDClassifier(loss='squared_hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=1000, tol=None, epsilon='huber')),])

text_clf.fit(X_train, y_train) 
    
predicted = text_clf.predict(X_test)
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