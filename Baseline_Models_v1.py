import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional, Conv1D,MaxPooling1D,Embedding, LSTM, Input, Flatten, Dense, Dropout, Convolution2D, Conv2D, MaxPooling2D, Lambda, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, AveragePooling2D, Concatenate
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn import decomposition, ensemble
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import re
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

# load the dataset
df= pd.read_csv('/Users/yuwenyu/Desktop/PhD./fake_news/Constraint_English_Train.csv')
df_val= pd.read_csv('/Users/yuwenyu/Desktop/PhD./fake_news/Constraint_English_Val.csv')
df_test= pd.read_csv('/Users/yuwenyu/Desktop/PhD./fake_news/Constraint_English_Test.csv')

df_all = pd.concat([df, df_val])
X_train=df_all['tweet']
y_train=df_all['label']
X_test= df_test['tweet']
y_test=df_test['label']

# encoding label
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test= encoder.fit_transform(y_test)

#NB classifier  + gridsearch
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
nb = Pipeline([('vect',CountVectorizer(analyzer='word',stop_words='english')),('tfidf',TfidfTransformer()),('clf', MultinomialNB(fit_prior=False)),])
nb_clf = nb.fit(X_train,y_train)

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1,1),(1,2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3)}
gs_clf = GridSearchCV(nb_clf, parameters, n_jobs=-1,cv=15)
gs_clf = gs_clf.fit(X_train,y_train)

from sklearn.metrics import f1_score,average_precision_score
predicted_nb = gs_clf.predict(df_test['tweet'])
accuracy_nb=round(np.mean(predicted_nb ==y_test),3)
print('the accuracy is ', accuracy_nb) 

from sklearn.metrics import classification_report
print(classification_report(y_test, predicted_nb, labels=[1,0], digits=4))


#svm + gridsearch
from sklearn.linear_model import SGDClassifier
svm = Pipeline([('vect',CountVectorizer(stop_words='english')),('tfidf',TfidfTransformer()),('clf-svm',SGDClassifier(loss='modified_huber',penalty='l2',alpha=0.001,shuffle=True)),])
svm_clf = svm.fit(X_train,y_train)

predicted_svm = svm_clf.predict(X_test)
print(classification_report(y_test, predicted_svm, labels=[1,0], digits=4))



