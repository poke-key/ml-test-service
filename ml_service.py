from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import json
import io

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Vercel

def predict_traffic(csv_data):
    try:
        # Load model and scaler
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'svm_tuned_model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')

        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'svm_tuned_model.pkl')
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return {'error': 'Model files not found. Please ensure svm_tuned_model.pkl and scaler.pkl are in the models directory.'}

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Read CSV data from string
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Map predictions to labels
        label_map = {0: "Streaming", 1: "Secure", 2: "DNS", 3: "Web", 4: "Other"}
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'index': i, 
                'category': int(pred), 
                'label': label_map.get(int(pred), f"Class {pred}")
            })
        
        # Calculate category counts
        counts = {}
        for result in results:
            label = result['label']
            counts[label] = counts.get(label, 0) + 1
        
        category_counts = [{'category': k, 'count': v} for k, v in counts.items()]
        
        return {
            'predictions': results, 
            'categoryCounts': category_counts, 
            'message': 'Predictions completed successfully'
        }
        
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get CSV data from request
        data = request.get_json()
        if not data or 'csv_data' not in data:
            return jsonify({'error': 'No CSV data provided'}), 400
        
        csv_data = data['csv_data']
        
        # Make predictions
        result = predict_traffic(csv_data)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Service error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'ML service is running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 