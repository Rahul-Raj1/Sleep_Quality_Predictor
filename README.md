Project Documentation: Sleep Quality Prediction using Machine Learning
1. Introduction

Sleep Quality Prediction is a machine learning classification system. It reads a CSV dataset of daily lifestyle habits (age, gender, screen time, caffeine intake and exercise hours), cleans the data, trains a Logistic Regression model, and predicts whether a person’s sleep quality is Good or Poor. A Streamlit application then shows the prediction as a sleep score and recommends personalised advice.

2. Problem Statement

Users should be able to enter their daily habits and see how likely they are to sleep well. The system should handle messy real-world data (missing values and duplicate rows) and turn the prediction into practical advice. It should not present itself as a medical tool, because it only learns patterns from the data it is given.

3. Objectives
Build a CSV-driven prediction system for sleep quality.
Clean the data using pandas (duplicates, null values, unused columns).
Use exploratory data analysis to understand how habits relate to sleep.
Train and evaluate a Logistic Regression model.
Check the model for overfitting and underfitting.
Provide a personalised advice engine based on the prediction.
Provide a clean Streamlit interface with a what-if simulator.
4. Technology Stack
Python
pandas and NumPy
scikit-learn (LogisticRegression, StandardScaler, Pipeline, metrics)
Matplotlib and Seaborn
joblib
Streamlit
Jupyter Notebook / VS Code
5. Dataset

The dataset sleep_quality_data.csv is synthetic and was generated with a Python script. The Good/Poor label comes from a hidden rule plus random noise: more screen time and caffeine reduce the chance of good sleep, and more exercise increases it.

Columns:

id
gender
age
screen_time
caffeine_intake
exercise_hours
sleep_quality (target)

Raw file: 445 rows, 25 duplicate rows and 64 missing values, added on purpose for preprocessing practice.
After cleaning: 420 rows (222 Poor and 198 Good) with no nulls and no duplicates.
Recommended final size for real use: a larger dataset collected from real people, for example through surveys or wearable devices.

6. Methodology

Step 1: Load data
Pandas reads the CSV file, and the shape, data types, null counts and duplicate counts are inspected.

Step 2: Clean data
The 25 duplicate rows are removed and the id column is dropped. Missing numeric values are filled with the median and missing gender values with the mode.

Step 3: Exploratory data analysis
Histograms, bar charts, boxplots and a correlation heatmap show the patterns. People with good sleep have lower screen time and caffeine intake and more exercise.

Step 4: Encode data
The target is mapped to Good = 1 and Poor = 0. Gender is one-hot encoded.

Step 5: Split data
The data is split 80:20 into 336 training rows and 84 testing rows, with stratification so both sets keep the same class balance.

Step 6: Train model
A scikit-learn Pipeline applies StandardScaler and then Logistic Regression. The scaler is fitted on training data only, which prevents data leakage.

Step 7: Evaluate model

Training accuracy: 74.1%
Testing accuracy: 82.1%
5-fold cross-validation accuracy: 74.5%
ROC-AUC: 0.88

A confusion matrix, classification report and learning curve confirm that the model is not overfitting.

Step 8: Predict and advise
The model returns the probability of good sleep. If it is at least 0.5, the result is Good, otherwise Poor. Rule-based advice is then chosen from the user’s screen time, caffeine, exercise and age. The model is saved as model.pkl and used by the Streamlit app.

7. Why Logistic Regression?

Logistic Regression is simple, fast and easy to explain, and it gives a probability as well as a class label. Its coefficients show which habit helps or hurts sleep. Here exercise has the strongest positive effect, followed by negative effects from screen time and caffeine.

Example:

Age 25, 2 hours of screen time, 1 cup of caffeine and 1.5 hours of exercise gives a 95.8% chance of good sleep.
Age 35, 9.5 hours of screen time, 6 cups of caffeine and 0.1 hours of exercise gives a 0.6% chance of good sleep.
8. User Interface

The Streamlit interface provides:

A header and four tabs: Predict, What-if Simulator, Sleep Tips and About.
Sliders for age, screen time, caffeine and exercise, and a gender selector.
A Good/Poor badge and a sleep score gauge from 0 to 100.
The probability of good and poor sleep.
A chart showing which habits affect the result.
Personalised advice cards.
A what-if simulator that shows the new score after habit changes.
9. Testing Strategy

Functional tests
Check healthy, unhealthy and borderline habit profiles, slider limits, tab switching and the error message shown when model.pkl is missing.

Dataset tests
Check that the required columns exist, that duplicates are removed (445 to 420 rows) and that no null values remain after cleaning.

Model tests
Check training and testing accuracy, cross-validation, the confusion matrix, ROC-AUC and the learning curve. A small gap between training and cross-validated accuracy means the model is not overfitting.

Threshold tuning
The default decision threshold is 0.5. Other values, such as 0.4 or 0.6, can be tried if one kind of error matters more. The threshold should be chosen using cross-validation, not only the result on one test split.

10. Limitations
The data is synthetic, so the results show the method rather than real sleep science.
Only five input features are used; stress, diet, sleep duration and health conditions are not included.
The target has only two classes.
Missing values were filled before the train/test split; strictly, the medians should be computed on the training set only.
The test set is small (84 rows), so cross-validation is more reliable.
The application is for education and is not medical advice.
11. Future Scope
Collect real data and add more features.
Compare Random Forest, Gradient Boosting and SVM with Logistic Regression.
Predict a sleep score or several categories instead of two classes.
Add user accounts and a database to track habits over time.
Add REST APIs using FastAPI.
Deploy the app to the cloud with monitoring.
Add AI-generated personalised coaching.
12. Conclusion

The project demonstrates a practical machine learning workflow from a messy CSV dataset to a deployed web application. It combines data cleaning, exploratory analysis, a Logistic Regression pipeline, evaluation with overfitting checks, a rule-based advice engine and a Streamlit interface that predicts sleep quality and recommends better habits.

I numbered the sections 1 to 12 in order. Your example skips from 6 to 9 because of the step numbering. If you want it to match exactly, I can restore those numbers.
