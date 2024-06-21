## Fake-News-Detection-Using-Machine-Learning

This project aims to tackle the growing problem of fake news by employing machine learning techniques. Our goal is to build a model that can automatically classify a piece of news as real or fake, helping to maintain the integrity of information circulated on various media platforms.
![image](https://github.com/jicsjitu/Fake_News_Using_ML/assets/162569175/47fab845-c272-4ee1-92b3-339b0956cc1a)

### Overview
The "Fake News Detection Using Machine Learning" project is designed to identify and categorize news articles into 'fake' or 'real'. Fake news can have significant adverse impacts on individuals and society. By leveraging natural language processing (NLP) and machine learning (ML) algorithms, this project seeks to develop a robust fake news detection system. We use datasets containing examples of real and fake news to train our models.
![image](https://github.com/jicsjitu/Fake_News_Using_ML/assets/162569175/aabae8a3-ee4e-47b8-8f1a-88659ed66ad0)

### Data Preprocessing

Data preprocessing is a critical step in any machine-learning workflow. In this project, I:

**Clean text data** by removing HTML tags, special characters, and stopwords to enhance the quality of the dataset.

**Normalize text** through stemming and lemmatization to reduce words to their base or root form.

**Vectorization** techniques such as TF-IDF are used to convert text to a format suitable for ML modeling.

#### Feature Engineering

Feature engineering involves extracting and selecting important variables based on the text data. We explore:

**N-grams,** which help in capturing the context of words in a given dataset.

**Sentiment analysis** to gauge the emotional tone behind a text.

**Word embeddings** like Word2Vec or GloVe to capture semantic meanings of words.

**User Interface:** A simple GUI for real-time news verification by users.

### The Web Interface

![WhatsApp Image 2024-06-06 at 17 25 44_acf837d9](https://github.com/jicsjitu/Fake_News_Using_ML/assets/162569175/ceffe157-c481-46ab-bf61-1adcdcd2bf1c)

### Dataset

The dataset includes labeled news articles with the following attributes:

Title: The title of the news article.

Text: The full text of the news article.

Label: 'FAKE' or 'REAL'.

The dataset used in this project can be found here: [Link](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) to Dataset

### Model Training and Evaluation

We experiment with various machine learning models to find the most effective in distinguishing fake news:

**Logistic Regression:** A baseline model for binary classification tasks.

**Support Vector Machine (SVM):** Useful for high-dimensional spaces like text data.

**Random Forest:** An ensemble method that can manage overfitting more effectively than decision trees.

**Neural Networks:** Deep learning models that can capture complex patterns in data.

**Decision Tree Classifier:** A decision tree classifier is a supervised machine learning technique that creates a tree-like model of choices and potential outcomes. 

**Gradient Boost Classifier:** Gradient Boosting is a functional gradient algorithm that repeatedly selects a function that leads in the direction of a weak hypothesis or negative gradient so that it can minimize a loss function. The gradient boosting classifier combines several weak learning models to produce a powerful predicting model.
Each model's performance is evaluated using metrics like accuracy, precision, recall, and F1-score to ensure robustness and reliability.

### Technologies Used

**Python:** Primary programming language

**NumPy & Pandas:** For data manipulation

**Scikit-learn:** For implementing machine learning algorithms

**NLTK:** For natural language processing tasks

**Streamlit:** For creating a web application to interact with the model

#### Author
Jitu Kumar
