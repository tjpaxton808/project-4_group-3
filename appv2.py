
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import os

# Set the template folder to templatesv2
app = Flask(__name__, template_folder='templatesv2')

# Load the headline model
with open('news_fake_real_model.pkl', 'rb') as f:
    headline_model = pickle.load(f)

# Load the webpage content model
with open('webpage_truth_model.pkl', 'rb') as f:
    webpage_model = pickle.load(f)


def predict_headline(headline, model):
    """
    Predict if a headline is fake or real news
    Args:
        headline (str): The news headline text
        model: The trained model pipeline
    Returns:
        prediction (int): 0 for fake, 1 for real
        probability (float): Probability of the prediction
    """
    prediction = model.predict([headline])[0]
    proba = model.predict_proba([headline])[0]
    probability = proba[1] if prediction == 1 else proba[0]
    return prediction, probability


def predict_webpage(text, model):
    """
    Predicts the truthfulness of a webpage text.
    Returns a tuple: (prediction, confidence, category)
    where prediction: 0 for false, 1 for true,
    confidence: probability associated with the predicted class,
    and category: 'News article' if prediction confidence is above threshold, else 'Possibly not a news article'
    """
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence = proba[1] if pred == 1 else proba[0]
    if max(proba) < 0.6:
        category = 'Possibly not a news article'
    else:
        category = 'News article'
    return pred, confidence, category


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_headline', methods=['POST'])
def predict_headline_route():
    headline = request.form['headline']
    prediction, probability = predict_headline(headline, headline_model)
    result = {
        'prediction': 'Real' if prediction == 1 else 'Fake',
        'probability': round(probability * 100, 2),
        'headline': headline
    }
    return jsonify(result)


@app.route('/predict_webpage', methods=['POST'])
def predict_webpage_route():
    webpage_text = request.form['webpage_text']
    prediction, confidence, category = predict_webpage(webpage_text, webpage_model)
    result = {
        'prediction': 'True' if prediction == 1 else 'False',
        'confidence': round(confidence * 100, 2),
        'category': category,
        'text_length': len(webpage_text)
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
