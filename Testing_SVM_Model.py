# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 12:28:45 2025

@author: bianc
"""

import pickle
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stop_words=set(stopwords.words('english'))

#---Loading test database---
phishing_db=pd.read_csv(r"C:\Users\bianc\Desktop\Proiect MCS\TREC_05_mic.csv")

#---Keep the most relevant columns---
phishing_db = phishing_db[['sender', 'subject', 'body', 'label']]


#---Test emails' preprocessing---
def preprocesare_text(text):
    if not isinstance(text, str):  # If text is not a string (e.g., NaN or float), convert it
        text = str(text) if text is not None else ""
    text=text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text
#---Columns concatenation and text preprocessing---
phishing_db['text_cleaned'] = phishing_db['sender'] + " " + phishing_db['subject'] + " " + phishing_db['body']
phishing_db['text_cleaned'] = phishing_db['text_cleaned'].apply(preprocesare_text)



#---Creating feature and label vectors---
X=phishing_db['text_cleaned']
y=phishing_db['label']


#---Text into numerical values conversion---
with open(r"C:\Users\bianc\Desktop\Proiect MCS\vectorizer_ana.pkl", 'rb') as f:
    vectorizer = pickle.load(f)

X_final_vect = vectorizer.transform(X).toarray()  # Convert to NumPy array


#---Load trained SVM model---
with open(r"C:\Users\bianc\Desktop\Proiect MCS\model_SVM_C=1_vectorizer_ana.pkl", 'rb') as f:
    clasificator_SVM = pickle.load(f)

y_prezis=clasificator_SVM.predict(X_final_vect)
#---Evaluation on new data---

acuratete_test=accuracy_score(y,y_prezis)
matrice_confuzie_test=confusion_matrix(y, y_prezis)

print(f"Acuratete: {acuratete_test:.4f}")
print("Matricea de confuzie:\n", matrice_confuzie_test)