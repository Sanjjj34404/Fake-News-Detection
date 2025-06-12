# 📰 Fake News Detection System

A machine learning-based application to detect fake and real news articles using Logistic Regression and TF-IDF vectorization. This project includes a full training pipeline and an interactive web interface built with Streamlit for real-time news verification.

## 🚀 Project Overview

The rapid spread of misinformation in digital media has made fake news detection a crucial challenge. This project implements a lightweight, interpretable, and effective solution that classifies news articles into *fake* or *real* categories based on textual content.

- **Algorithm:** Logistic Regression
- **Feature Extraction:** TF-IDF (with bi-grams)
- **Accuracy:** ~94% on test data
- **Frontend:** Streamlit-based web interface
- **Output:** CSV files of predictions + visualizations


## 🧪 Features

- Preprocessing with NLTK (stopword removal, tokenization, regex cleaning)
- TF-IDF feature extraction with unigrams and bigrams
- Logistic Regression model training with scikit-learn
- Evaluation metrics: Accuracy, Precision, Recall, F1-score
- Visualization: Confusion matrix, prediction distribution
- Top keyword extraction for model explainability
- Real-time prediction web app using Streamlit


## 📁 Datasets

- True.csv: Contains real news articles
- Fake.csv: Contains fake news articles
- test.csv: Unlabeled dataset for prediction

Dataset source: Kaggle Fake News Dataset


