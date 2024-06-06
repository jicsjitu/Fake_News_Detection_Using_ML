import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st
import requests
from bs4 import BeautifulSoup

# Streamlit page configuration
st.set_page_config(page_title="Fake News Detector", layout="wide")

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load and process data
@st.cache_data
def load_and_process_data(sample_size=1000):
    dataframe_fake = pd.read_csv("Fake.csv")
    dataframe_true = pd.read_csv("True.csv")
    dataframe_fake['label'] = 1
    dataframe_true['label'] = 0
    news_dataframe = pd.concat([dataframe_fake.sample(sample_size), dataframe_true.sample(sample_size)], axis=0).reset_index(drop=True)
    return news_dataframe

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
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(news_dataframe['content'])
    y = news_dataframe['label'].values
    return X, y, vectorizer

# Train model
@st.cache_data
def train_model(_X, y):
    X_train, X_test, y_train, y_test = train_test_split(_X, y, test_size=0.2, stratify=y, random_state=2)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

# Function to scrape news articles from a website
def scrape_news_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('article')
    news_data = []
    for article in articles:
        title = article.find('h2').text.strip()
        content = article.find('div', class_='article-content').text.strip()
        news_data.append({'title': title, 'content': content})
    return news_data

# Function to classify news articles using the trained model
def classify_news_articles(articles):
    for article in articles:
        text = article['content']
        preprocessed_text = stem_content(text)
        vectorized_text = vectorizer.transform([preprocessed_text])
        prediction = model.predict(vectorized_text)[0]
        confidence = model.predict_proba(vectorized_text)[0][prediction]
        article['prediction'] = prediction
        article['confidence'] = confidence

# Load data and train model
news_dataframe = load_and_process_data()
X, y, vectorizer = vectorize_data(news_dataframe)
model, accuracy = train_model(X, y)

# Streamlit app
st.title('📰 Fake News Detector')
st.write("""
    Welcome to the Fake News Detector! This application uses Natural Language Processing (NLP) techniques to classify news articles as real or fake. 
    Simply enter the news text in the box below or upload a text file and click on 'Predict' to see the result.
""")

st.sidebar.title("🛠 App Options")
if st.sidebar.button("Reload Data"):
    st.cache_data.clear()
    st.experimental_rerun()

st.sidebar.markdown("### Model Performance")
st.sidebar.write(f"Accuracy: {accuracy:.2%}")

# Input form
input_text = ""
uploaded_file = None
with st.form(key='news_form'):
    input_text = st.text_area('Enter News Article', height=250)
    uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])
    if uploaded_file is not None:
        input_text = uploaded_file.read().decode("utf-8")
        st.text_area('File Content', input_text, height=250)
    submit_button = st.form_submit_button(label='Predict')
    clear_button = st.form_submit_button(label='Clear Input')

# If clear button is clicked, clear the input fields
if clear_button:
    input_text = ""
    uploaded_file = None

# Only perform prediction when submit button is clicked
if submit_button:
    if input_text:
        with st.spinner('Analyzing the article...'):
            input_data = vectorizer.transform([input_text])
            pred = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data)[0][pred]
            result = 'The News is Fake' if pred == 1 else 'The News is Real'
            st.success(result)
            st.info(f"Confidence: {confidence:.2%}")
            st.write("Key Features:")
            feature_names = vectorizer.get_feature_names_out()
            coefficients = model.coef_[0]
            sorted_indices = coefficients.argsort()
            top_features = [feature_names[i] for i in sorted_indices[:10]] if pred == 1 else [feature_names[i] for i in sorted_indices[-10:]]
            st.write(top_features)
    else:
        st.error("Please enter some text to analyze.")

st.sidebar.markdown("### About")
st.sidebar.info("""
    **Developer:** Jitu Kumar  
    **University:** Usha Martin University, Ranchi Jharkhand  

    This application is built using Streamlit and applies Logistic Regression on TF-IDF features extracted from the news text.  

    **Data Source:** [Kaggle Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
""")

# Real-time News Scraping Section
st.title("🌐 Real-time News Scraping")
url_input = st.text_input("Enter the URL of a news website to scrape:")
if st.button("Scrape News"):
    if url_input:
        try:
            st.info("Scraping news articles from the provided URL...")
            news_articles = scrape_news_website(url_input)
            if news_articles:
                classify_news_articles(news_articles)
                st.subheader("Classified News Articles:")
                for article in news_articles:
                    title = article['title']
                    prediction = 'Fake' if article['prediction'] == 1 else 'Real'
                    confidence = article['confidence']
                    st.write(f"Title: {title}")
                    st.write(f"Prediction: {prediction} (Confidence: {confidence:.2%})")
                    st.write("---")
            else:
                st.warning("No articles found on the provided URL.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
