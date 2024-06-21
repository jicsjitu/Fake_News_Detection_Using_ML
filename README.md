## Fake-News-Detection-Using-Machine-Learning

This project aims to tackle the growing problem of fake news by employing machine learning techniques. Our goal is to build a model that can automatically classify a piece of news as real or fake, helping to maintain the integrity of information circulated on various media platforms.
![31876582-removebg-preview](https://github.com/jicsjitu/Fake-News-Detection-Using-Machine-Learning/assets/162569175/ab553405-6465-4506-8fd1-b63a05f5a943)
### Overview
The "Fake News Detection Using Machine Learning" project is designed to identify and categorize news articles into 'fake' or 'real'. Fake news can have significant adverse impacts on individuals and society. By leveraging natural language processing (NLP) and machine learning (ML) algorithms, this project seeks to develop a robust fake news detection system. We use datasets containing examples of real and fake news to train our models.

### Data Preprocessing

Data preprocessing is a critical step in any machine-learning workflow. In this project, we:

**Clean text data** by removing HTML tags, special characters, and stopwords to enhance the quality of the dataset.

**Normalize text** through stemming and lemmatization to reduce words to their base or root form.

**Vectorization** techniques such as TF-IDF are used to convert text to a format suitable for ML modeling.

![image](https://github.com/jicsjitu/Fake-News-Detection-Using-Machine-Learning/assets/162569175/f790ac66-a744-45b8-888f-df4146b1f187)

#### Feature Engineering

Feature engineering involves extracting and selecting important variables based on the text data. We explore:

**N-grams,** which help in capturing the context of words in a given dataset.

**Sentiment analysis** to gauge the emotional tone behind a text.

**Word embeddings** like Word2Vec or GloVe to capture semantic meanings of words.

**User Interface:** A simple GUI for real-time news verification by users.

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

**Flask:** For creating a web application to interact with the model

#### Images :

![image](https://github.com/jicsjitu/Fake-News-Detection-Using-Machine-Learning/assets/162569175/582b36f3-cfad-4c56-b9c8-05bebff1fc38) 

![image](https://github.com/jicsjitu/Fake-News-Detection-Using-Machine-Learning/assets/162569175/063a80e0-3e4f-4757-af3e-956927310ebc)



![image](https://github.com/jicsjitu/Fake-News-Detection-Using-Machine-Learning/assets/162569175/730f98de-69dc-446e-b6eb-0ea03607f665)
                                          
#### Author
Jitu Kumar
