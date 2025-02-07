from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# # Load the dataset
# data = pd.read_csv('mental_health_data.csv')  # Replace 'mental_health_data.csv' with your actual dataset file

# # Separate features (X) and target variable (y)
# X = data[['Age', 'Gender', 'Family_History']]
# y = data['Mental_Treatment']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the Logistic Regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Get the accuracy of the model
# accuracy = accuracy_score(y_test, model.predict(X_test))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    family_history = int(request.form['family_history'])

    # Create a DataFrame with the user's input
    user_data = pd.DataFrame({'Age': [age], 'Gender': [gender], 'Family_History': [family_history]})

    # Predict the probability of requiring mental treatment
    # probability = model.predict_proba(user_data)[0][1]

    return f'The probability of requiring mental treatment is: test'

if __name__ == '__main__':
    app.run(debug=True)