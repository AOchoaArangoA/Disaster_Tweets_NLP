# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:39:09 2022

@author: Administrador
"""

#%%
import pandas as pd 
import numpy as np 
import re
import string #Evaluar que es cada libreria
import nltk #Libreria para el tema de las palabras
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier ,AdaBoostClassifier ,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB ,MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
#%%

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#%%
train = train[['text', 'target']].copy()
test = test[['text', 'id']].copy()

#1. Limpiar la informacion
#2. Procesamiento de la informacion
#3. Estimar y evaluar

def Nulos(df):
    for i in df.columns: 
        print('Col: {} / Num Nan: {}'.format(i, df[i].isnull().sum()))

Nulos(train)
print('---------------')
Nulos(test)

#%%

#Primero pasamos todo a minuscula 

train['Limpia'] =  train["text"].apply(lambda x: x.lower())
print(train.at[0, 'text'])
print(train.at[0, 'Limpia'])

#%% 
def remove_URL(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)
train['Limpia'] = train['Limpia'].apply(lambda x: remove_URL(x))
def remove_html(text):
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)
train['Limpia'] = train['Limpia'].apply(lambda x: remove_html(x))
def remove_emojis(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
train['Limpia'] = train['Limpia'].apply(lambda x: remove_emojis(x))
def remove_punct(text):
    return text.translate(str.maketrans('', '', string.punctuation))
train['Limpia'] = train['Limpia'].apply(lambda x: remove_punct(x))

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7f]',r'', text)
train['Limpia'] = train['Limpia'].apply(lambda x: remove_non_ascii(x))
#%%

#Hace parte del preprocesamiento de la informacion
train['token'] = train['Limpia'].apply(lambda x: x.split())

stop = set(stopwords.words('english'))
train['stop_word'] = train['token'].apply(lambda x: [word for word in x if word not in stop])

def porter_stemmer(text):
    stemmer = nltk.PorterStemmer()
    stems = [stemmer.stem(i) for i in text]
    return stems

train['abreviatura'] = train['stop_word'].apply(lambda x: porter_stemmer(x))

#%%
nltk.download('wordnet')
def lemmatize_word(text):
    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(word) for word in text]
    return lemma
train['lemma'] = train['stop_word'].apply(lambda x: lemmatize_word(x))
train['final_txt']=train['lemma'].apply(lambda x: ''.join(i+' ' for i in x))
#%%

#Luego del preprocesamiento, se realiza la vectorizacion de los datos
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer='word', ngram_range = (1,1))
x_tr = vectorizer.fit_transform(train['final_txt'])

columns= vectorizer.get_feature_names()
data = x_tr.toarray()
print(data.shape)

#%%
y = train['target']
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.5, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)
y_hat =model.predict(X_test)
print(classification_report(y_test,y_hat))

#%%
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_hat =model.predict(X_test)
print(classification_report(y_test,y_hat))





