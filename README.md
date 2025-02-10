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

### 2. Doctor-Patient Conversation Analysis (NLP) - Healthcare Emergency Alert System

This project analyzes doctor-patient conversations using NLP to extract insights, detect emergency cases, and classify conversations into different risk levels. The aim is to identify symptoms, assess patient distress, and flag high-risk cases that need immediate attention.

#### **Key Components**

- **Data Preprocessing & Cleaning:** Removing noise (special characters, numbers, stopwords) and standardizing text using lemmatization.
- **Exploratory Data Analysis (EDA):** Identifying common symptoms, treatment discussions, and emotional trends.
- **Sentiment Analysis:** Classifying patient conversations as positive, neutral, or negative using TextBlob/VADER.
- **Symptom and Risk Classification:** Training a model (Random Forest, CountVectorizer) to classify conversations into low-risk and high-risk based on symptoms and distress levels.
- **Emergency Detection System:** Detecting critical symptoms (e.g., severe pain, fatigue, dizziness, bleeding, tumor growth) and psychological distress (e.g., anxiety, hopelessness, fear) using keyword detection and NLP models.
- **Interactive Dashboard for Insights:** Developing a dashboard (plotly/Dash) to visualize findings and flag emergency cases.
- **Calander Intgration with cal.com** A unique meeting link is generated.
 Patients can visit the link to customize their meeting schedule.
 During the booking process, they are required to provide their name and email.
 Upon successful booking, they receive a confirmation email with meeting details.



#### **AI-Powered Healthcare Risk Detection**
My solution uses AI-powered risk detection and real-time monitoring to assess patient symptoms remotely. Patients fill out an online questionnaire, and our system analyzes their responses to detect potential health risks. If a high risk is found, the system alerts doctors immediately, allowing for quick medical response. This improves patient safety while reducing unnecessary hospital visits.

**Workflow**
Patient Dashboard – Collects patient information and assesses health risks.
Doctor’s Dashboard – Organizes patient data into a table for further analysis and medical decisions.
Visualization Board – Analyzes dynamic patient data for trends and insights.


#### **Technologies Used**

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- NLTK, TextBlob
- Scikit-learn (sklearn)
- WordCloud
- Dash/plotly (for visualization)
- Google Colab (for training)
- ngrok (for running the dashboard)
- streamlit
- joblib 

---

## Expected Outcomes

✅ A structured and cleaned dataset from cancer patient conversations
✅ EDA insights on symptoms, treatments, and emotional impact
✅ A sentiment classification model to understand patient distress
✅ A trained ML model to classify conversations into low-risk and high-risk
✅ An emergency detection system that flags urgent cases
✅ An interactive dashboard to visualize findings
✅ An integration cal.com to schedule a Meeting with Doctor

## Installation & Usage

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk textblob wordcloud dash plotly gspread
```


! Feel free to submit issues or pull requests for improvements.

## License

This project is licensed under the MIT License.




