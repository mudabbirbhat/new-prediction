from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Include this for CORS support
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and label encoder
with open('model_RandomForest.pkl', 'rb') as f:
    model, label_encoder = pickle.load(f)

# Define a root route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML file

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the JSON data sent in the POST request
    
    # Assuming the input data is sent as a list of features
    features = np.array([data['features']])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Convert numerical prediction to the disorder label
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    # Return the result as a JSON response
    return jsonify({
        'prediction': predicted_label
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
