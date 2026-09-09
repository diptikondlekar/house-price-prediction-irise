import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Safely resolve model path relative to app execution directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear.pkl')
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background-color: #f3f4f6;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 24px 16px;
        }

        .container {
            background: #ffffff;
            border-radius: 12px;
            padding: 32px;
            max-width: 550px;
            width: 100%;
            /* Drop Shadows */
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            border: 1px solid #e5e7eb;
        }

        h1 {
            color: #111827;
            font-size: 22px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 6px;
        }

        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 24px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .full-width {
            grid-column: span 2;
        }

        label {
            font-size: 13px;
            font-weight: 600;
            color: #374151;
            margin-bottom: 6px;
        }

        input[type="number"], select {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
            background-color: #ffffff;
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05);
        }

        input[type="number"]:focus, select:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
        }

        .submit-btn {
            grid-column: span 2;
            margin-top: 8px;
            padding: 12px;
            background-color: #2563eb;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
        }

        .submit-btn:hover {
            background-color: #1d4ed8;
            box-shadow: 0 6px 12px -2px rgba(37, 99, 235, 0.4);
        }

        .result-card {
            margin-top: 24px;
            padding: 16px;
            border-radius: 8px;
            background-color: #eff6ff;
            border: 1px solid #bfdbfe;
            text-align: center;
            color: #1e40af;
            font-size: 18px;
            font-weight: 700;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
    </style>
</head>
<body>

<div class="container">
    <h1>House Price Predictor</h1>
    <p class="subtitle">Enter property details based on dataset specifications</p>

    <form action="/predict" method="post" class="form-grid">
        
        <div class="form-group">
            <label for="square_footage">Square Footage</label>
            <input type="number" step="any" id="square_footage" name="square_footage" placeholder="e.g. 1360" required>
        </div>

        <div class="form-group">
            <label for="num_bedrooms">Bedrooms</label>
            <input type="number" id="num_bedrooms" name="num_bedrooms" placeholder="e.g. 2" required>
        </div>

        <div class="form-group">
            <label for="num_bathrooms">Bathrooms</label>
            <input type="number" id="num_bathrooms" name="num_bathrooms" placeholder="e.g. 1" required>
        </div>

        <div class="form-group">
            <label for="year_built">Year Built</label>
            <input type="number" id="year_built" name="year_built" placeholder="e.g. 1981" required>
        </div>

        <div class="form-group">
            <label for="lot_size">Lot Size (Acres)</label>
            <input type="number" step="any" id="lot_size" name="lot_size" placeholder="e.g. 0.60" required>
        </div>

        <!-- Categorical Field: Garage Size -->
        <div class="form-group">
            <label for="garage_size">Garage Size</label>
            <select id="garage_size" name="garage_size" required>
                <option value="" disabled selected>Select option</option>
                <option value="0">0 (No Garage)</option>
                <option value="1">1 (1-Car/Attached)</option>
                <option value="2">2 (2-Car Garage)</option>
                <option value="3">3 (3-Car Garage)</option>
            </select>
        </div>

        <!-- Categorical Field: Neighborhood Quality -->
        <div class="form-group full-width">
            <label for="neighborhood_quality">Neighborhood Quality (1-10)</label>
            <select id="neighborhood_quality" name="neighborhood_quality" required>
                <option value="" disabled selected>Select quality score</option>
                <option value="1">1 - Very Low</option>
                <option value="2">2 - Low</option>
                <option value="3">3 - Below Average</option>
                <option value="4">4 - Fair</option>
                <option value="5">5 - Average</option>
                <option value="6">6 - Above Average</option>
                <option value="7">7 - Good</option>
                <option value="8">8 - Very Good</option>
                <option value="9">9 - Excellent</option>
                <option value="10">10 - Premium</option>
            </select>
        </div>

        <button type="submit" class="submit-btn">Predict House Price</button>
    </form>

    {% if prediction_text %}
        <div class="result-card">
            {{ prediction_text }}
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction_text="Error: linear.pkl file not found.")

    try:
        # Extract features matching the exact feature order[cite: 1]
        sqft = float(request.form['square_footage'])
        bedrooms = float(request.form['num_bedrooms'])
        bathrooms = float(request.form['num_bathrooms'])
        year = float(request.form['year_built'])
        lot_size = float(request.form['lot_size'])
        garage_size = float(request.form['garage_size'])
        neighborhood_quality = float(request.form['neighborhood_quality'])

        # Arrange array according to X = df[['Square_Footage','Num_Bedrooms','Num_Bathrooms','Year_Built','Lot_Size','Garage_Size','Neighborhood_Quality']][cite: 1]
        features = np.array([[sqft, bedrooms, bathrooms, year, lot_size, garage_size, neighborhood_quality]])
        prediction = model.predict(features)[0]

        result = f"Estimated Price: ${prediction:,.2f}"
    except Exception as e:
        result = f"Error during evaluation: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
