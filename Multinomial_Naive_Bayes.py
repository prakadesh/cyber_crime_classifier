import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from joblib import dump, load
from flask import Flask, request, jsonify

# Load the data
data = pd.read_csv("cybercrime.csv")

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Extract the features and labels
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_data['message'])
train_labels = train_data['label']
test_features = vectorizer.transform(test_data['message'])
test_labels = test_data['label']

# Train the models
model1 = MultinomialNB()
model2 = LinearSVC(random_state=42)

model1.fit(train_features, train_labels)
model2.fit(train_features, train_labels)

# Save the trained models
dump(model1, 'model1.joblib')
dump(model2, 'model2.joblib')
dump(vectorizer, 'vectorizer.joblib')

# Load the trained models
model1 = load('model1.joblib')
model2 = load('model2.joblib')
vectorizer = load('vectorizer.joblib')

# Create a Flask app
app = Flask(__name__)

# Define an API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    message = request.json['message']
    input_features = vectorizer.transform([message])

    pred1 = model1.predict(input_features)
    pred2 = model2.predict(input_features)

    results = {
        'prediction_1': pred1[0],
        'prediction_2': pred2[0],
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
