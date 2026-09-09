import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load trained linear regression model
model = pickle.load(open('linear.pkl', 'rb'))

# HTML template with modern layout and shadow effects inside app.py
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
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: #f4f7f6;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            background: #ffffff;
            border-radius: 12px;
            padding: 30px 40px;
            max-width: 600px;
            width: 100%;
            /* Shadow Effects */
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08), 0 4px 10px rgba(0, 0, 0, 0.05);
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 25px;
            font-size: 26px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        label {
            font-size: 14px;
            color: #555;
            margin-bottom: 6px;
            font-weight: 600;
        }

        input[type="number"], select {
            padding: 10px 12px;
            border: 1px solid #ccc;
            border-radius: 6px;
            font-size: 14px;
            transition: all 0.3s ease;
            /* Box Shadow for Inputs */
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
        }

        input[type="number"]:focus, select:focus {
            border-color: #4A90E2;
            outline: none;
            box-shadow: 0 0 8px rgba(74, 144, 226, 0.3);
        }

        .submit-btn {
            margin-top: 20px;
            grid-column: span 2;
            padding: 12px;
            background-color: #4A90E2;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4);
        }

        .submit-btn:hover {
            background-color: #357ABD;
            box-shadow: 0 6px 15px rgba(74, 144, 226, 0.5);
            transform: translateY(-1px);
        }

        .result-box {
            margin-top: 25px;
            padding: 15px;
            background-color: #eaf4fe;
            border-left: 5px solid #4A90E2;
            border-radius: 4px;
            text-align: center;
            font-size: 18px;
            color: #1c3d5a;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
        }
    </style>
</head>
<body>

<div class="container">
    <h1>🏠 House Price Prediction</h1>
    <form action="/predict" method="post" class="form-grid">
        
        <div class="form-group">
            <label for="square_footage">Square Footage</label>
            <input type="number" step="any" id="square_footage" name="square_footage" placeholder="e.g. 1500" required>
        </div>

        <div class="form-group">
            <label for="num_bedrooms">Bedrooms</label>
            <input type="number" id="num_bedrooms" name="num_bedrooms" placeholder="e.g. 3" required>
        </div>

        <div class="form-group">
            <label for="num_bathrooms">Bathrooms</label>
            <input type="number" id="num_bathrooms" name="num_bathrooms" placeholder="e.g. 2" required>
        </div>

        <div class="form-group">
            <label for="year_built">Year Built</label>
            <input type="number" id="year_built" name="year_built" placeholder="e.g. 2005" required>
        </div>

        <div class="form-group">
            <label for="lot_size">Lot Size (Acres)</label>
            <input type="number" step="any" id="lot_size" name="lot_size" placeholder="e.g. 1.2" required>
        </div>

        <!-- Categorical Column: Garage Size -->
        <div class="form-group">
            <label for="garage_size">Garage Size</label>
            <select id="garage_size" name="garage_size" required>
                <option value="" disabled selected>Select garage status</option>
                <option value="0">No Garage (0)</option>
                <option value="1">Has Garage (1)</option>
            </select>
        </div>

        <!-- Categorical Column: Neighborhood Quality -->
        <div class="form-group full-width">
            <label for="neighborhood_quality">Neighborhood Quality Rating</label>
            <select id="neighborhood_quality" name="neighborhood_quality" required>
                <option value="" disabled selected>Select rating category</option>
                <option value="1">1 - Poor</option>
                <option value="2">2 - Low</option>
                <option value="3">3 - Fair</option>
                <option value="4">4 - Below Average</option>
                <option value="5">5 - Average</option>
                <option value="6">6 - Above Average</option>
                <option value="7">7 - Good</option>
                <option value="8">8 - Very Good</option>
                <option value="9">9 - Excellent</option>
                <option value="10">10 - Luxury</option>
            </select>
        </div>

        <button type="submit" class="submit-btn">Predict House Price</button>
    </form>

    {% if prediction_text %}
        <div class="result-box">
            <strong>{{ prediction_text }}</strong>
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
    # Extract numerical and categorical inputs from the form
    sqft = float(request.form['square_footage'])
    bedrooms = float(request.form['num_bedrooms'])
    bathrooms = float(request.form['num_bathrooms'])
    year = float(request.form['year_built'])
    lot_size = float(request.form['lot_size'])
    garage_size = float(request.form['garage_size'])
    neighborhood_quality = float(request.form['neighborhood_quality'])

    # Prepare features array for prediction matching training dataset features order
    features = np.array([[sqft, bedrooms, bathrooms, year, lot_size, garage_size, neighborhood_quality]])
    
    # Predict price using linear regression model
    prediction = model.predict(features)[0]
    
    result_text = f"Estimated House Price: ${prediction:,.2f}"
    
    return render_template_string(HTML_TEMPLATE, prediction_text=result_text)

if __name__ == '__main__':
    app.run(debug=True)
