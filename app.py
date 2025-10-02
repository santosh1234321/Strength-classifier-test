from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import joblib
import os
from werkzeug.exceptions import BadRequest

# --- Configuration ---
MODEL_FILE = 'strength_model.pkl'
SCALER_FILE = 'scaler.pkl'
# REQUIRED_KEYS now reflects the new feature set (weights assumed in KG)
REQUIRED_KEYS = ['bench', 'squat', 'deadlift', 'gender', 'pull ups', 'push ups']

app = Flask(__name__)
CORS(app) # Initialize CORS to allow cross-origin requests (e.g., from frontend on a different port)

# --- Load Model and Scaler ---
try:
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        raise FileNotFoundError(f"Missing required files ({MODEL_FILE} and/or {SCALER_FILE}). Please run train_model.py first.")
        
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print(f"Model ({MODEL_FILE}) and Scaler ({SCALER_FILE}) loaded successfully.")

except Exception as e:
    print(f"ERROR: Failed to load ML artifacts: {e}")
    model = None 
    scaler = None

@app.route('/', methods=['GET'])
def home():
    """Simple health check and instruction page."""
    status = "Ready" if model and scaler else "ERROR: Model/Scaler Missing"
    return jsonify({
        'status': status,
        'message': 'ML Strength Classifier API is running.',
        'instruction': 'POST JSON data to /predict',
        'required_data_keys': REQUIRED_KEYS,
        'units_note': 'Weights (bench, squat, deadlift) must be provided in Kilograms (KG).'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests based on the new feature set."""
    if not model or not scaler:
        return jsonify({'error': 'Prediction service is unavailable. Model or scaler artifact is missing.'}), 503

    try:
        data = request.get_json(force=True)
        
        # Input validation: check for required keys
        if not all(key in data for key in REQUIRED_KEYS):
            missing_keys = [key for key in REQUIRED_KEYS if key not in data]
            raise BadRequest(f"Missing required keys in JSON payload: {', '.join(missing_keys)}")

        # Extract features in the correct order used during training:
        # bench (KG), squat (KG), deadlift (KG), gender (0=male, 1=female), pull ups, push ups
        features = np.array([
            data['bench'],
            data['squat'],
            data['deadlift'],
            data['gender'], 
            data['pull ups'],
            data['push ups']
        ]).reshape(1, -1)

        # Scale features using the saved scaler object
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        
        result_label = 'strong' if prediction == 1 else 'not strong'
        
        return jsonify({
            'prediction': result_label,
            'status': 'success',
            'input_data': data
        })

    except BadRequest as e:
        return jsonify({'error': str(e.description)}), 400
    except Exception as e:
        # Catch any unexpected errors during scaling or prediction
        return jsonify({'error': f"An internal error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Running on 0.0.0.0 makes the app externally accessible, which is sometimes necessary 
    # when the frontend and backend are served differently locally (like your 5500 vs 5000 ports).
    app.run(debug=True, port=5000, host='0.0.0.0')
