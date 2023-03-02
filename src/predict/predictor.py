import pickle
from flask import Flask, jsonify, request
import numpy as np
import json

app = Flask(__name__)

# Load the trained scikit-learn models stored in pickle format
with open('../../models/travelModel.pkl', 'rb') as f:
    modelo_tiempo_viaje = pickle.load(f)

with open('../../models/deliveryModel.pkl', 'rb') as f:
    modelo_tiempo_entrega = pickle.load(f)

with open('../../models/le.pkl', 'rb') as f:
    labelEncoder = pickle.load(f)



# Endpoint for route prediction model
# Input is a json object with attribute time
@app.route('/predict_eta', methods=['POST'])
def predict_eta():

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediccion[0]})

# Endpoint for load delivery endpoint.
# Input is a json object with attributes truckId and time
@app.route('/predict_delivery', methods=['POST'])
def predict_delivery():
        


    # Return the prediction as a JSON response  
    return jsonify({'prediction': prediccion[0]})


if __name__ == '__main__':
    app.run(debug=True, port =7777)