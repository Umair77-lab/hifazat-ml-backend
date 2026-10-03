from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

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

        # Map purity estimate based on ML severity classification
        if severity_label == 'Safe':
            purity = 98
        elif severity_label == 'Water Diluted':
            purity = 75
        else: # Chemically Adulterated / Powder(highly unsafe)
            purity = 10

        return jsonify({
            'status': 'success',
            'severity': severity_label,
            'confidence': round(confidence, 1),
            'purity_percent': purity
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    # Run locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)