from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app) # CORS successfully enabled for Chrome testing

# Load machine learning artifacts
model = joblib.load('hifazat_model.pkl')
scaler = joblib.load('hifazat_scaler.pkl')
encoder = joblib.load('hifazat_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features in exact trained order
        features = np.array([[
            float(data.get('pH_Raw', 6.7)),
            float(data.get('TDS_Raw', 0)),
            float(data.get('Turbidity_Raw', 0)),
            float(data.get('Color_R', 0)),
            float(data.get('Color_G', 0)),
            float(data.get('Color_B', 0)),
            float(data.get('Temperature_C', 25.0)),
            float(data.get('MQ135_Raw', 0)),
            float(data.get('MQ3_Raw', 0))
        ]])

        # Scale features and run prediction
        scaled_features = scaler.transform(features)
        pred_idx = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        severity_label = encoder.inverse_transform([pred_idx])[0]
        confidence = float(np.max(probabilities) * 100)

        # --- FIX: Map the ML label to Flutter's expected UI Severity ---
        if severity_label == 'Safe':
            frontend_severity = 'safe'
            purity = 98
        elif severity_label == 'Water Diluted':
            frontend_severity = 'caution' # Triggers Yellow Warning in App
            purity = 75
        else: # Chemically Adulterated / Powder (highly unsafe)
            frontend_severity = 'unsafe'  # Triggers Red Alert in App
            purity = 10

        return jsonify({
            'status': 'success',
            'severity': frontend_severity, # Sends exactly what Flutter expects
            'ml_label': severity_label,    # Keeps original for records
            'confidence': round(confidence, 1),
            'purity_percent': purity
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    # Run locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
