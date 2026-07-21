import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained pickle model
try:
    with open('linear_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Embedded HTML Template with Glassmorphism and CSS Animations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            min-height: 100vh;
            background: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #020617);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 40px 20px;
            color: #f8fafc;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.125);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 30px;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a855f7, #6366f1, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        header p {
            color: #94a3b8;
            font-size: 0.95rem;
        }

        .result-card {
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(99, 102, 241, 0.2));
            border: 1px solid rgba(168, 85, 247, 0.4);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            margin-bottom: 30px;
            animation: pulse 2s infinite alternate;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 15px rgba(168, 85, 247, 0.2); }
            100% { box-shadow: 0 0 30px rgba(168, 85, 247, 0.5); }
        }

        .result-card h2 {
            font-size: 1.1rem;
            color: #cbd5e1;
            margin-bottom: 5px;
        }

        .result-card .price {
            font-size: 2.5rem;
            font-weight: 700;
            color: #38bdf8;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        .input-group label {
            font-size: 0.85rem;
            color: #cbd5e1;
            margin-bottom: 6px;
            font-weight: 600;
        }

        .input-group input {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            padding: 12px 14px;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus {
            border-color: #a855f7;
            box-shadow: 0 0 10px rgba(168, 85, 247, 0.3);
            background: rgba(15, 23, 42, 0.8);
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 15px;
            padding: 16px;
            background: linear-gradient(135deg, #a855f7, #6366f1);
            color: #fff;
            border: none;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(168, 85, 247, 0.4);
        }

        .submit-btn:active {
            transform: translateY(0);
        }
    </style>
</head>
<body>

    <div class="container">
        <header>
            <h1>House Valuation AI</h1>
            <p>Enter property specifications below for an instant estimated price</p>
        </header>

        {% if prediction %}
        <div class="result-card">
            <h2>Estimated Property Value</h2>
            <div class="price">${{ prediction }}</div>
        </div>
        {% endif %}

        <form method="POST" action="/" class="grid-form">
            {% for feature in features %}
            <div class="input-group">
                <label for="{{ feature.key }}">{{ feature.label }}</label>
                <input type="number" step="any" id="{{ feature.key }}" name="{{ feature.key }}" 
                       placeholder="{{ feature.placeholder }}" required 
                       value="{{ form_data.get(feature.key, '') }}">
            </div>
            {% endfor %}

            <button type="submit" class="submit-btn">Calculate Value</button>
        </form>
    </div>

</body>
</html>
"""

# Feature metadata extracted from your linear_model.pkl
FEATURES = [
    {"key": "bedrooms", "label": "Number of Bedrooms", "placeholder": "e.g. 3"},
    {"key": "bathrooms", "label": "Number of Bathrooms", "placeholder": "e.g. 2.5"},
    {"key": "living_area", "label": "Living Area (sq ft)", "placeholder": "e.g. 2000"},
    {"key": "lot_area", "label": "Lot Area (sq ft)", "placeholder": "e.g. 5000"},
    {"key": "floors", "label": "Number of Floors", "placeholder": "e.g. 1.5"},
    {"key": "waterfront", "label": "Waterfront Present (0 or 1)", "placeholder": "0 or 1"},
    {"key": "views", "label": "Number of Views", "placeholder": "0 to 4"},
    {"key": "condition", "label": "House Condition", "placeholder": "1 to 5"},
    {"key": "grade", "label": "House Grade", "placeholder": "1 to 13"},
    {"key": "area_no_basement", "label": "Area (Excl. Basement)", "placeholder": "e.g. 1500"},
    {"key": "basement_area", "label": "Area of Basement", "placeholder": "e.g. 500"},
    {"key": "built_year", "label": "Built Year", "placeholder": "e.g. 1995"},
    {"key": "renovation_year", "label": "Renovation Year", "placeholder": "e.g. 2010 (0 if none)"},
    {"key": "lot_area_renov", "label": "Lot Area Renovated", "placeholder": "e.g. 5000"},
    {"key": "schools", "label": "Schools Nearby", "placeholder": "e.g. 3"},
    {"key": "airport_dist", "label": "Distance from Airport (km)", "placeholder": "e.g. 12.5"}
]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    form_data = {}

    if request.method == 'POST':
        try:
            form_data = request.form
            # Extract inputs in the exact order expected by linear_model.pkl
            input_values = [float(request.form.get(f['key'], 0)) for f in FEATURES]
            
            if model:
                features_array = np.array([input_values])
                pred = model.predict(features_array)[0]
                prediction_text = f"{max(0, pred):,.2f}"
            else:
                prediction_text = "Error: Model file not loaded properly."
        except Exception as e:
            prediction_text = f"Invalid input format: {e}"

    return render_template_string(
        HTML_TEMPLATE, 
        features=FEATURES, 
        prediction=prediction_text, 
        form_data=form_data
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
