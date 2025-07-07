# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 12:06:59 2025

@author: bianc
"""
#---Loading necessary libraries---

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# ---Downloading stopwords---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---Loading the database---
phishing_db = pd.read_csv(r"C:\Users\bianc\Desktop\Proiect MCS\baza_de_date_finala.csv")


# ---Text preprocessing function---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

#---Columns concatenation---
phishing_db['text_cleaned'] = phishing_db['sender'] + " " + phishing_db['subject'] + " " + phishing_db['body']
#---Actual text preprocessing---
phishing_db['text_cleaned'] = phishing_db['text_cleaned'].apply(preprocess_text)

#---Defining feature and label vectors---
X = phishing_db['text_cleaned']
y = phishing_db['label']

#---Transform text in numerical values---
vectorizer = TfidfVectorizer(stop_words="english",
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2)
X_tfidf = vectorizer.fit_transform(X).toarray()  # Convert to NumPy array

#---Save vectorizer---
import pickle
with open(r'C:\Users\bianc\Desktop\Proiect MCS\vectorizer_ana.pkl','wb') as f:
    pickle.dump(vectorizer,f)



# ---Splitting the dataset---
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)



#---Model initialization---

svm_model = SVC(kernel='rbf',C=1,gamma='scale')

#---Model training---
svm_model.fit(X_train, y_train)

#---Predictions on test set---
y_pred = svm_model.predict(X_test)

#---Model evaluation---
acuratete = accuracy_score(y_test, y_pred)
matrice_confuzie = confusion_matrix(y_test, y_pred)

print(f"Acuratete antrenare: {acuratete:.4f}")
print("Matricea de confuzie:\n", matrice_confuzie)


#---Save model---
import pickle
with open(r'C:\Users\bianc\Desktop\Proiect MCS\model_SVM_C=1_vectorizer_ana.pkl','wb') as f:
    pickle.dump(svm_model,f)

