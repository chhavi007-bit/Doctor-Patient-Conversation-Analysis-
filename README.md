# Cancer Patient Analysis & Doctor-Patient Conversation Analysis

## Overview

This repository contains two machine learning projects focused on cancer patient analysis and doctor-patient conversation classification. The goal is to leverage structured health data and Natural Language Processing (NLP) techniques to improve patient risk assessment and emergency detection.

## Projects

### 1. Cancer Patient Analysis (KNN & Random Forest)

This project applies classification techniques using K-Nearest Neighbors (KNN) and Random Forest (RF) to analyze cancer patient data. The dataset includes 23 health-related features to predict patient risk levels (High, Medium, Low). Hyperparameter tuning and cross-validation are used to optimize model performance.

#### **Dataset**

The dataset consists of 1000 patient records with the following features:

- **Demographics:** Age, Gender
- **Health Factors:** Smoking, Alcohol Use, Obesity, Genetic Risk
- **Symptoms:** Chest Pain, Fatigue, Weight Loss, Coughing Blood
- **Target Variable:** Level (Converted to `Level_numeric`: High → 1, Medium → 2, Low → 3)

#### **Models Implemented**

- K-Nearest Neighbors (KNN)
- Random Forest (RF) Classifier

#### **Technologies Used**

- Python
- Pandas, NumPy
- Scikit-Learn (sklearn)
- Matplotlib, Seaborn (for visualization)
- Google Sheets API (gspread)

---

### 2. Doctor-Patient Conversation Analysis (NLP)

This project analyzes doctor-patient conversations using NLP to extract insights, detect emergency cases, and classify conversations into different risk levels. The aim is to identify symptoms, assess patient distress, and flag high-risk cases that need immediate attention.

#### **Key Components**

- **Data Preprocessing & Cleaning:** Removing noise (special characters, numbers, stopwords) and standardizing text using lemmatization.
- **Exploratory Data Analysis (EDA):** Identifying common symptoms, treatment discussions, and emotional trends.
- **Sentiment Analysis:** Classifying patient conversations as positive, neutral, or negative using TextBlob/VADER.
- **Symptom and Risk Classification:** Training a model (Random Forest, CountVectorizer) to classify conversations into low-risk and high-risk based on symptoms and distress levels.
- **Emergency Detection System:** Detecting critical symptoms (e.g., severe pain, fatigue, dizziness, bleeding, tumor growth) and psychological distress (e.g., anxiety, hopelessness, fear) using keyword detection and NLP models.
- **Interactive Dashboard for Insights:** Developing a dashboard (Streamlit/Dash) to visualize findings and flag emergency cases.
![Alt Text](Capture.png)

#### **Technologies Used**

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- NLTK, TextBlob
- Scikit-learn (sklearn)
- WordCloud
- Dash/Streamlit (for visualization)
- Google Colab (for training)
- ngrok (for running the dashboard)

---

## Expected Outcomes

✅ A structured and cleaned dataset from cancer patient conversations
✅ EDA insights on symptoms, treatments, and emotional impact
✅ A sentiment classification model to understand patient distress
✅ A trained ML model to classify conversations into low-risk and high-risk
✅ An emergency detection system that flags urgent cases
✅ An interactive dashboard to visualize findings

## Installation & Usage

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk textblob wordcloud dash streamlit gspread
```

### Running the Models

1. **Cancer Patient Analysis:**
   ```bash
   python cancer_analysis.py
   ```
2. **Doctor-Patient Conversation Analysis:**
   ```bash
   python nlp_analysis.py
   ```

### Running the Dashboard

```bash
python run dashboard.py
```



! Feel free to submit issues or pull requests for improvements.

## License

This project is licensed under the MIT License.




