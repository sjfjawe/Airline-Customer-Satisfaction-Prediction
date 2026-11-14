from flask import Flask, request, jsonify
import pickle
from predict import load_model, predict

app = Flask(__name__)

model, dv, scaler = load_model()

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    result = predict(data, model, dv, scaler)
    return jsonify(result)

@app.route('/', methods=['GET'])
def home():
    return "Airline Satisfaction Prediction API"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
