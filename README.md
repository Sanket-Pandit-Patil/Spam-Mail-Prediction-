# Spam-Mail-Prediction

This project demonstrates the use of machine learning techniques to classify emails as either spam or non-spam (ham). It leverages a Logistic Regression model and the TF-IDF Vectorizer to extract meaningful features from text data for effective classification.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Workflow](#project-workflow)
6. [Results](#results)
7. [Technologies Used](#technologies-used)

---

## Overview

Spam emails are a significant nuisance and a potential security risk. This project builds a predictive model to identify spam emails with high accuracy, using a dataset of labeled emails.

---

## Features

- Preprocessing of raw email data to clean and standardize text.
- Extraction of meaningful features using TF-IDF Vectorization.
- Training and testing of a Logistic Regression model for classification.
- Performance evaluation using metrics like accuracy, precision, recall, and F1-score.

---

## Installation

### Setup on Google Colab
1. Open Google Colab: [Google Colab](https://colab.research.google.com).
2. Upload the Jupyter notebook file (`spam_mail_prediction.ipynb`) to your Colab environment.
3. Ensure the required libraries are available in the Colab runtime. Colab typically includes most required libraries like `scikit-learn`, `pandas`, and `numpy` pre-installed.
4. If needed, install additional dependencies in the Colab notebook using the following command:
   
   ```python
   !pip install <library_name>
---
## Usage :   
Upload Dataset: Upload your email dataset (e.g., a CSV file) to the Colab notebook. You can do this by clicking the "Files" tab in the Colab sidebar and using the upload option.  
Run Notebook: Execute the cells in the notebook sequentially to preprocess the data, train the model, and evaluate the performance.  
View Output: The notebook will display the output metrics and plots to help you understand the model's effectiveness.  

---

## Project Workflow :  
Data Preprocessing:  
Clean and tokenize the email text.  
Remove stop words, punctuation, and other irrelevant text elements.  
  
Feature Extraction :  
Apply TF-IDF Vectorization to convert text into numerical features that can be used by machine learning models.  
  
Model Training and Testing :  
Train a Logistic Regression model using the transformed data.  
Test the model with unseen email data to evaluate its classification performance.  
  
Evaluation :  
Use metrics like accuracy, precision, recall, and F1-score to assess how well the model identifies spam and non-spam emails.  

---

## Technologies Used
Programming Language : Python  

Libraries:  
scikit-learn: For machine learning algorithms and utilities.  
NumPy: For numerical operations and data handling.  
Pandas: For data manipulation and handling datasets.  
TF-IDF Vectorizer: To convert text data into numerical features for the model.  

---

## Results :  
The model achieves high accuracy in distinguishing spam from non-spam emails, making it suitable for practical use in email filtering systems. You can evaluate the performance of the model with the following metrics:

Accuracy: Measures the percentage of correct predictions.  
Precision: Measures how many of the predicted spam emails were actually spam.  
Recall: Measures how many of the actual spam emails were correctly identified.  
F1-score: The harmonic mean of precision and recall, giving a balanced view of the model's performance.  


