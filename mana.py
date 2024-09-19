from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('solar_power_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # Data in JSON format
    prediction = model.predict(np.array(data).reshape(1, -1))
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
