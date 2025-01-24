# Doctor-Patient-Conversation-Analysis-
This project focuses on analyzing doctor-patient conversations using Natural Language Processing (NLP) techniques to extract meaningful insights, detect emergency cases, and classify conversations into different risk levels. The goal is to  detect symptoms, and flag high-risk cases that require immediate attention.

1. Data Preprocessing & Cleaning:
- Removing noise (special characters, numbers, stopwords) and standardizing text through lemmatization.
Exploratory Data Analysis (EDA):
- Identifying common symptoms, treatment discussions, and emotional trends in patient conversations.
Sentiment Analysis:
- Categorizing patient conversations as positive, neutral, or negative using TextBlob/VADER to assess emotional impact.
Symptom and Risk Classification:
- Training a machine learning model (Random Forest, Logistic Regression, countvector) to classify conversations into low-risk and high-risk categories based on symptoms and distress levels.
Emergency Detection System:
- Identifying critical symptoms (e.g., severe pain, fatigue, dizziness, bleeding, tumor growth) and psychological distress (e.g., anxiety, hopelessness, fear) using keyword detection and NLP models.
Interactive Dashboard for Insights:
- Developing a dashboard (Streamlit/Dash) to visualize key findings, trends, and flagged emergency cases.

2. Programming Language: Python
Libraries & Tools: Pandas, NumPy, Matplotlib, Seaborn, NLTK, TextBlob, Scikit-learn, WordCloud, Dash/Streamlit
Machine Learning Models: Random Forest, Logistic Regression, CountVectorizer (for text vectorization)
Deployment: Google Colab (for training) and ngrok (for running the dashboard)

3. Expected Outcomes
✅ A structured and cleaned dataset from cancer patient conversations
✅ EDA insights on symptoms, treatments, and emotional impact
✅ A sentiment classification model to understand patient distress
✅ A trained ML model to classify conversations into low-risk and high-risk
✅ An emergency detection system that flags urgent cases
✅ An interactive dashboard to visualize findings

# cancer patient analysis
This model is designed for classification tasks using K-Nearest Neighbors (KNN) and Random Forest, leveraging structured data from Google Sheets. It aims to categorize entities based on extracted features and optimize performance using hyperparameter tuning and cross-validation.
This project analyzed a dataset containing 23 health-related features of cancer patients. It explores different machine learning techniques to predict risk levels (High, Medium, Low) and optimize model accuracy.


1. Models Implemented:
K-Nearest Neighbors (KNN)
Random Forest (RF) Classifier

2. Dataset
The dataset contains 1000 patient records with features such as:
Demographics: Age, Gender
Health Factors: Smoking, Alcohol Use, Obesity, Genetic Risk
Symptoms: Chest Pain, Fatigue, Weight Loss, Coughing Blood
Target Variable: Level (Converted to Level_numeric: High → 1, Medium → 2, Low → 3)
3. Technologies Used
Python
Pandas, NumPy
Scikit-Learn (sklearn)
Matplotlib, Seaborn (for visualization)
Google Sheets API (gspread)
