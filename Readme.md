# Email Text Analysis and Classification

This project focuses on analyzing and classifying email text data using natural language processing (NLP) techniques and machine learning models. The main tasks include data preprocessing, feature extraction, sentiment analysis, and email classification.

## Overview

The Python script (`email_analysis_classification.py`) demonstrates several key steps:

1. **Data Loading and Preprocessing:**
   - Loads the email dataset (`emails.csv`) using pandas.
   - Cleans the email text by removing HTML tags, special characters, and emojis.
   - Splits the dataset into training and validation sets.

2. **Text Cleaning:**
   - Defines functions to clean text by removing HTML tags and emojis.
   - Uses regular expressions and libraries like BeautifulSoup and emoji for text cleaning.

3. **Stemming and Lemmatization:**
   - Applies stemming and lemmatization techniques using NLTK (PorterStemmer, WordNetLemmatizer) and spaCy.
   - Demonstrates how to generate stemmed and lemmatized versions of email bodies.

4. **Feature Extraction:**
   - Extracts TF-IDF (Term Frequency-Inverse Document Frequency) features from the cleaned email text using sklearn's TfidfVectorizer.
   - Implements Bag-of-Words (BoW) representation for feature extraction.

5. **Model Training and Evaluation:**
   - Trains two types of classifiers: Naive Bayes and Logistic Regression.
   - Evaluates model performance using metrics such as accuracy, precision, recall, and F1-score.
   - Visualizes confusion matrix and histograms for positive and negative word counts.

6. **Feature Engineering:**
   - Extracts handcrafted features such as text length, average word length, capital letters, punctuation count, and numeral count from email text.
   - Calculates total counts and displays descriptive statistics.

7. **Sentiment Analysis:**
   - Performs sentiment analysis using NLTK's SentimentIntensityAnalyzer.
   - Counts positive and negative words in email text and visualizes their distributions.

8. **Regularization Techniques:**
   - Implements Logistic Regression with L1 and L2 regularization (using sklearn's LogisticRegression).

## Dependencies

Ensure you have the following dependencies installed:

- Python 3.x
- pandas
- scikit-learn
- nltk
- spacy
- emoji

Install dependencies using pip:

```bash
pip install pandas scikit-learn nltk spacy emoji
