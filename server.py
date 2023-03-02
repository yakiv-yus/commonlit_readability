import pandas as pd
import numpy as np
import pickle
from flask import Flask, jsonify, request
import requests
import json

with open('text_transformer.pkl', 'rb') as f:
    text_transformer = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Convert the data to a numpy array and transform text to features
    data = np.array(data['data'])
    data =  text_transformer.transform(data)

    # Make a prediction
    prediction = model.predict(data)

    # Convert the prediction to a list
    prediction = prediction.tolist()

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

app.run(port=5000)
