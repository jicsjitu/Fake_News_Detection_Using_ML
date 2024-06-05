import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Load and process data
@st.cache_data
def load_and_process_data():
    dataframe_fake = pd.read_csv("Fake.csv")
    dataframe_true = pd.read_csv("True.csv")
    dataframe_fake['label'] = 1
    dataframe_true['label'] = 0
    news_dataframe = pd.concat([dataframe_fake, dataframe_true], axis=0).reset_index(drop=True)
    return news_dataframe

# Stem content
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
def stem_content(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

# Vectorize data
@st.cache_data
def vectorize_data(news_dataframe):
    news_dataframe['content'] = news_dataframe['text'].apply(stem_content)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(news_dataframe['content'])
    y = news_dataframe['label'].values
    return X, y, vectorizer

# Train model
@st.cache_data
def train_model(_X, _y):
    X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.2, stratify=_y, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Load data and train model
news_dataframe = load_and_process_data()
X, y, vectorizer = vectorize_data(news_dataframe)
model = train_model(X, y)

# Streamlit app
st.title('Fake News Detector')
input_text = st.text_input('Enter news article')

if input_text:
    input_data = vectorizer.transform([input_text])
    pred = model.predict(input_data)[0]
    st.write('The News is Fake' if pred == 1 else 'The News Is Real')
