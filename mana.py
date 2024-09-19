from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('solar_power_model.pkl')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json['data']
        
        # Convert data to a NumPy array and make prediction
        prediction = model.predict(np.array(data).reshape(1, -1))
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Main entry point to run the app
if __name__ == '__main__':
    # Bind to 0.0.0.0 and the appropriate port
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
