# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:31:19 2022

@author: Viren Fernandes
"""

import json
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('wordnet') # used the wordnet corpus from the NLTK package
nltk.download('averaged_perceptron_tagger')

#Opening JSON file
# x = open('train_others.json')
# y = open('dev.json')
z = open('train_spider.json') 
 
#Loading JSON File
# train_data = json.load(x)
# col_data = json.load(y)
dev_data = json.load(z)

#Segregating features and labels
features = []
labels = []

#Function to word on the lemmatizing the tokenised word
def lemmatizer(text):
    lemmatizer=WordNetLemmatizer()
    lemm_text = [lemmatizer.lemmatize(word) for word in result]
    return lemm_text

#loop for preprocessing 
for data in dev_data:         
    #Lower() - change the text to lowercase
    #strip() - remove whitespaces in the sentences
    #replace() - remove question mark from the text.
    result = data['question'].lower().strip().replace("?","") 
    
    #word tokenise
    result = word_tokenize(result)   
    
    #Lematizing of the result
    lemma = lemmatizer(result)
    #NLTK POS Tagging
    POS = pos_tag(result)
    
    features.append(POS)
    
    labels.append(data['query_toks'])

X = features #features
#Features Example for 1st element: [('how', 'WRB'), ('many', 'JJ'), ('heads', 'NNS'), ('of', 'IN'), ('the', 'DT'), ('departments', 'NNS'), ('are', 'VBP'), ('older', 'JJR'), ('than', 'IN'), ('56', 'CD')]
y = labels #labels
#Labels Example for 1st element : ['SELECT', 'count', '(', '*', ')', 'FROM', 'head', 'WHERE', 'age', '>', '56']

# for Representation purpose displayed the first element in the list of the features and labels
print('Features = ',X[0]) #English text
print('Labels = ',y[0]) #tokenised query






