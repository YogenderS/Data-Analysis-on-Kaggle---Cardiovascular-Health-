# Data-Analysis-on-Kaggle---Cardiovascular-Health-
Cardiovascular illnesses (CVDs) are the major cause of death worldwide. CVDs include coronary heart disease, cerebrovascular disease, rheumatic heart disease, and other heart and blood vessel problems. According to the World Health Organization, 17.9 million people die each year. Heart attacks and strokes account for more than four out of every five CVD deaths, with one-third of these deaths occurring before the age of 70. A comprehensive database for factors that contribute to a heart attack has been constructed.

# Problem Statement
The main purpose here is to collect characteristics of Heart Attack or factors that contribute to it. The size of the dataset is 1319 samples, which have nine fields, where eight fields are for input fields and one field for an output field. Age, gender(0 for Female, 1 for Male) ,heart rate (impulse), systolic BP (pressurehight), diastolic BP (pressurelow), blood sugar(glucose), CK-MB (kcm), and Test-Troponin (troponin) are representing the input fields, while the output field pertains to the presence of heart attack (class), which is divided into two categories (negative and positive); negative refers to the absence of a heart attack, while positive refers to the presence of a heart attack.

# 🛠 Tools Used
Tools used: 

Google Colab (library: Seaborn, Matplotlib, Pandas, Numpy, sklearn and scipy)

Excel

# Documentation
# Python Analysis:-
# Table of Contents:

1️⃣ Import Libraries:
Import essential Python libraries such as pandas, numpy, scikit-learn, and matplotlib.

2️⃣ Load Dataset:
Load the dataset into a Pandas DataFrame, using functions like pd.read_csv() or pd.read_excel().

3️⃣ Exploratory Data Analysis:
Examine the first few rows of the dataset using df.head() to get a sense of the data.
Check data types, statistics, and summary statistics using functions like df.info(), df.describe().
Visualize data with plots (histograms, scatter plots, etc.) to understand the distribution of variables using libraries like Matplotlib and Seaborn.
Identify any trends, patterns, or outliers in the data.

4️⃣ Checking Null, Duplicate, Outliers Values, Cleaning Data, Removing Noise from Data:
Check for missing values using df.isnull().sum() and handle them by imputing or dropping rows/columns as needed.
Identify and handle duplicate values using df.duplicated() and df.drop_duplicates().
Detect and address outliers using statistical methods or visualization techniques.
Clean and preprocess the data by handling categorical variables, scaling features, and addressing any data quality issues.

5️⃣ Feature Engineering and Building ML Models (Random Forest, Naive Bayes, KNN):
Feature engineering involves creating new features or transforming existing ones to improve model performance.
Split the dataset into training and testing sets using train_test_split.
Build and train machine learning models:
Random Forest: Create an instance of RandomForestClassifier or RandomForestRegressor and fit it to the training data.
Naive Bayes: Use GaussianNB or other variants depending on the problem type (classification/regression).
K-Nearest Neighbors (KNN): Instantiate KNeighborsClassifier or KNeighborsRegressor and train it on the training data.
Evaluate model performance with appropriate metrics (e.g., accuracy, F1-score, mean squared error) using cross-validation or test data.

6️⃣ Feature Importance, Insights, and Recommendations:
Determine feature importance using model-specific attributes (e.g., feature_importances_ for Random Forest).
Gain insights into which features have the most impact on the model's predictions.
Provide recommendations or actionable insights based on the analysis, such as which features to focus on for improvement or which model performed the best.

# Here are some insights and recommendations based on these results:

# Insights:

1️⃣ Feature Importance: Both the domain knowledge and the decision tree model highlight "kcm" and "troponin" as critical features for predicting heart attacks. This suggests that these features contain valuable information for diagnosis.

2️⃣ Model Performance: The Random Forest model outperforms the other two models significantly, with an accuracy of 97.98%. This indicates that Random Forest is the most suitable algorithm for this task, given the provided data.

3️⃣ Naive Bayes: Although Naive Bayes shows a relatively high accuracy of 94.44%, it lags behind Random Forest. This might be due to the model's assumption of independence among features, which may not hold true for your data.

4️⃣ K-Nearest Neighbors (KNN): KNN has the lowest accuracy at 62.37%, which suggests that it might not be the best choice for this problem. This may be due to the sensitivity of KNN to noisy data and the curse of dimensionality.

# Recommendations:

1️⃣ Focus on "kcm" and "troponin": Given their high importance in predicting heart attacks, medical professionals and researchers should pay special attention to the "kcm" and "troponin" values when assessing a patient's risk. These factors could be critical for early detection and intervention.

2️⃣ Data Quality: Since the model's performance heavily relies on the quality of data, ensure that the data collection and preprocessing steps are accurate and comprehensive. Outliers and data noise can impact the effectiveness of machine learning models.

3️⃣ Further Investigation: It's essential to delve deeper into the dataset to understand the relationships between these features and the occurrence of heart attacks. Investigate how "kcm" and "troponin" interact with other features to gain a more comprehensive understanding of the problem.

4️⃣ Consult Domain Experts: Collaboration with medical professionals and domain experts is crucial for interpreting model results and implementing recommendations effectively. Their expertise can provide valuable context and insights.


