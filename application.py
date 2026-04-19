from flask import Flask, request, jsonify
import joblib

# CRITICAL: AWS Elastic Beanstalk looks for the variable named 'application' 
application = Flask(__name__)

# Load the trained model into memory
model = joblib.load('sentiment_model.joblib')

# Update the decorator to use 'application' [cite: 233, 234]
@application.route('/predict', methods=['POST'])
def predict():
    # Parse the incoming JSON request
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided. Please send a JSON with a "text" key.'}), 400

    # Make a prediction
    prediction = model.predict([text])[0]
    
    # Return the result
    return jsonify({
        'input_text': text,
        'sentiment_prediction': prediction,
        'model_version': '1.1'
    })

if __name__ == '__main__':
    # For AWS, the simple run() command is sufficient [cite: 256]
    application.run()
