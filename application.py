from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model into memory
model = joblib.load('sentiment_model.joblib')

@app.route('/predict', methods=['POST'])
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
        'sentiment_prediction': prediction
    })

if __name__ == '__main__':
    # Listen on all network interfaces (0.0.0.0) so Docker can expose it
    app.run(host='0.0.0.0', port=5000)
