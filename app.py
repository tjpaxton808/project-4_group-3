from flask import Flask, request, jsonify, render_template  
import joblib  
import re  
import numpy as np  
   
app = Flask(__name__)  

@app.route('/')  
def home():  
    return render_template('index.html') 
   
# Load both models and vectorizers  
# News vs Non-News classifier  
news_classifier_vectorizer = joblib.load('model/tfidf_vectorizer_news_classifier_new.pkl')  
news_classifier_model = joblib.load('model/logistic_news_classifier_new.pkl') 
    
# Real vs Fake News classifier  
fake_news_vectorizer = joblib.load('model/tfidf_vectorizer_modified.pkl')  
fake_news_model = joblib.load('model/logistic_model_modified.pkl')  
   
def preprocess_text(text):  
    text = text.lower()  
    text = re.sub(r'[^\\w\\s]', ' ', text)  
    text = re.sub(r'\\s+', ' ', text)  
    return text.strip()  
     
@app.route('/predict', methods=['POST'])  
def predict():  
    try:  
        # Get text from request  
        if request.is_json:  
            data = request.json  
            text = data.get('text', '')  
        else:  
            text = request.form.get('text', '')  
           
        if not text:  
            return jsonify({'error': 'No text provided'})  
           
        # Step 1: Check if it's news or non-news  
        processed_text = preprocess_text(text)  
        news_features = news_classifier_vectorizer.transform([processed_text])  
        is_news_prediction = news_classifier_model.predict(news_features)[0]  
        news_proba = news_classifier_model.predict_proba(news_features)[0]  
           
        # Safeguard probabilities against NaN  
        news_proba = np.nan_to_num(news_proba)
           
        # Initialize variables  
        fake_probability = 0.0  
        real_probability = 0.0  
           
        # FIXED: Correctly interpret the prediction
        # In the model, 1 = news, 0 = non-news (Amazon descriptions)
        if is_news_prediction == 1:  # If predicted as news (class 1)
            # It's news, check if real or fake  
            fake_features = fake_news_vectorizer.transform([processed_text])  
            fake_proba = fake_news_model.predict_proba(fake_features)[0]  
            fake_proba = np.nan_to_num(fake_proba)  
               
            # Assuming index 1 is real news probability  
            real_probability = float(fake_proba[1])  
            fake_probability = float(fake_proba[0])  # or 1 - real_probability  
               
            if real_probability >= 0.7:  
                prediction = "REAL"  
                confidence = real_probability  
            else:  
                prediction = "FAKE"  
                confidence = fake_probability  
        else:  # If predicted as non-news (class 0)
            # It's not news  
            prediction = "NON-NEWS"  
            confidence = float(news_proba[0])  # Confidence it's non-news (class 0)
           
        # Return the format expected by your frontend  
        return jsonify({  
            'prediction': prediction,  
            'confidence': confidence,  
            'fake_probability': fake_probability,  
            'real_probability': real_probability,
            'is_news_prediction': int(is_news_prediction),  # Added for debugging
            'news_proba': news_proba.tolist()  # Added for debugging
        })  
           
    except Exception as e:  
        print(f"Error in prediction: {str(e)}")  
        return jsonify({'error': str(e)}), 500  
   
if __name__ == '__main__':  
    app.run(debug=True)  


# Write the modified app.py to a new file
with open('modified_app.py', 'w') as file:
    file.write(modified_app_content)

