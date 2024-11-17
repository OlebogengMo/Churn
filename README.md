**Customer Churn Prediction Using Machine Learning**

This project is a data science pipeline aimed at predicting customer churn using the Telco Customer Churn dataset. The project leverages Python's data analysis and machine learning libraries, such as pandas, numpy, matplotlib, seaborn, and scikit-learn, and implements a Random Forest Classifier to model churn.

Table of Contents
Overview
Dataset
Data Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering
Model Training
Model Evaluation
Project Structure
How to Run
Results
Future Improvements
License
Overview
Customer churn is a significant issue in the telecommunications industry, impacting a company's profitability. This project aims to predict whether a customer will churn (leave the company) based on several features, such as tenure, monthly charges, contract type, and more. The objective is to use machine learning to assist in identifying at-risk customers and improve retention strategies.

**Dataset**

The dataset used for this project is the Telco Customer Churn dataset, which contains information about 7,043 customers of a telecommunications company. Key features include:

Customer demographics: Gender, SeniorCitizen, Partner, Dependents
Account information: Tenure, Contract, Payment method, MonthlyCharges, TotalCharges
Services signed up for: Phone service, Internet service, Streaming services, etc.
Target variable: Churn (Yes/No)
Source
The dataset can be found on Kaggle.

**Data Preprocessing**

Removed unnecessary columns (e.g., customerID).
Converted TotalCharges to a numeric data type, handling non-numeric values by coercing them to NaN.
Addressed missing values by dropping rows with NaN values.
Encoded categorical features using LabelEncoder.
Standardized numerical features using StandardScaler.
Exploratory Data Analysis (EDA)
Visualized the distribution of churn using a count plot.
Plotted histograms of numerical features (tenure, MonthlyCharges, TotalCharges) to observe their distributions.
Created a correlation heatmap to identify relationships between features.
Feature Engineering
Encoded categorical features to numeric values using LabelEncoder.
Applied feature scaling to numerical features using StandardScaler.

**Project Structure**

├── README.md
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── customer_churn_prediction.ipynb
├── images
│   ├── churn_distribution.png
│   ├── histograms.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── requirements.txt

**How to Run**

Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/customer-churn-prediction.git
Navigate to the project directory:
bash
Copy code
cd customer-churn-prediction
Install the required packages:
Copy code
pip install -r requirements.txt
Open and run the Jupyter Notebook:
Copy code
jupyter notebook customer_churn_prediction.ipynb

**Results**

Classification Report: Shows precision, recall, and F1-score for each class.
Confusion Matrix: Visualizes the performance of the model in classifying churn and non-churn customers.
ROC-AUC Score: Evaluates the model's ability to distinguish between the classes.
Feature Importance: Highlights the most influential features for predicting churn.

**Future Improvements**

Experiment with other machine learning models, such as XGBoost or Logistic Regression, to compare performance.
Use hyperparameter tuning (e.g., Grid Search) to optimize the Random Forest model.
Address class imbalance using techniques like SMOTE or class weighting.
Incorporate feature selection techniques to simplify the model and improve interpretability.
Model Training
Split the data into training and testing sets using an 80-20 split.
Trained a Random Forest Classifier with 100 estimators to predict churn.
Model Evaluation
Evaluated the model using a classification report, confusion matrix, and ROC-AUC score.
Visualized the confusion matrix to show true positives, true negatives, false positives, and false negatives.
Computed and plotted feature importance to understand the contribution of each feature to the model.
