# -Mental-Health-in-Tech-Survey-arogo
Self-Analysis Mental Health Model
Introduction
Mental health is a crucial aspect of overall well-being, yet it often goes unnoticed and untreated. This project aims to predict potential mental health conditions using machine learning based on survey responses. By leveraging data-driven insights, the model helps identify individuals who may need mental health support, thereby contributing to early intervention and awareness.

Dataset
The dataset used in this project consists of survey responses related to mental health factors. It includes various demographic and workplace-related features such as age, gender, family history of mental health issues, treatment status, work interference, access to mental health benefits, available care options, anonymity concerns, and leave policies. These features help in understanding the potential indicators of mental health conditions.

Data Preprocessing
To ensure the dataset is clean and suitable for model training, several preprocessing steps were undertaken:

Handling Missing Values: Columns with significant missing values were removed, while others were filled with appropriate default values.
Encoding Categorical Variables: Categorical data, such as gender and treatment status, was converted into numerical values for compatibility with machine learning models.
Cleaning Inconsistencies: Inconsistent entries, particularly in categorical features like gender, were standardized.
Model Training
To determine the best-performing model for mental health prediction, multiple machine learning algorithms were trained and evaluated:

Logistic Regression
K-Nearest Neighbors (KNN)
Random Forest (Best Model)
Gaussian Naive Bayes
After testing different models, the Random Forest Classifier was selected as the best-performing model due to its high accuracy and ability to handle complex relationships within the data.

Model Evaluation
The trained models were evaluated using standard performance metrics, including accuracy, precision, recall, F1-score, and ROC AUC score. The Random Forest model achieved the highest accuracy of approximately 79.37%, outperforming the other models. This result indicates that the model can reliably classify whether an individual might require mental health support.

Flask Application
To make the model easily accessible to users, a Flask web application was developed. The application provides a user-friendly interface where individuals can input relevant details, such as age, work-related factors, and personal history. Based on these inputs, the trained model predicts whether the individual may need mental health support. The web application serves as a practical tool for increasing awareness and encouraging early intervention.

Conclusion
This project successfully demonstrates the application of machine learning in mental health prediction. By analyzing survey data, the model helps identify individuals who may require mental health support. The Random Forest model was found to be the most effective, achieving an accuracy of 79.37%. Additionally, the Flask web application enhances accessibility, allowing users to obtain predictions



