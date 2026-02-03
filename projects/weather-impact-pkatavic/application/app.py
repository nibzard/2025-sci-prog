"""
Flask Backend for Energy Consumption Predictor
PJM Energy Dataset (Pennsylvania) 2002-2018
LightGBM Model with R² = 0.94
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import math
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load model and metadata on startup
print("Loading model and metadata...")
with open('lgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

FEATURE_COLUMNS = model_info['feature_columns']
FEATURE_MEANS = model_info['feature_means']
TARGET_MEAN = model_info['target_mean']
TARGET_STD = model_info['target_std']
R2_SCORE = model_info['r2_score']

print(f"Model loaded successfully! R² = {R2_SCORE:.4f}")
print(f"Features: {len(FEATURE_COLUMNS)}")


def create_features(user_input):
    """
    Create all 50 features from user input, using means for missing values.
    """
    # Start with all means
    features = FEATURE_MEANS.copy()

    # Extract user inputs
    temperature = user_input.get('temperature', FEATURE_MEANS['apparent_temperature_min'])
    wind_speed = user_input.get('wind_speed', FEATURE_MEANS['wind_speed_10m_mean'])
    humidity = user_input.get('humidity', FEATURE_MEANS['relative_humidity_2m_mean'])
    is_weekend = user_input.get('is_weekend', 0)
    season = user_input.get('season', 'spring')
    date_str = user_input.get('date', None)

    # Parse date if provided
    if date_str:
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            year = date_obj.year
            dayofweek = date_obj.weekday()
            day_of_year = date_obj.timetuple().tm_yday
        except:
            year = 2018
            dayofweek = 2
            day_of_year = 180
    else:
        year = 2018
        dayofweek = 2
        day_of_year = 180

    # Update base features from user input
    features['apparent_temperature_min'] = temperature
    features['wind_speed_10m_mean'] = wind_speed
    features['wind_speed_10m_max'] = wind_speed * 1.5  # Estimate max
    features['relative_humidity_2m_mean'] = humidity
    features['year'] = year
    features['dayofweek'] = dayofweek
    features['is_weekend'] = 1 if is_weekend else 0

    # Season flags (one-hot)
    features['is_winter'] = 1 if season == 'winter' else 0
    features['is_spring'] = 1 if season == 'spring' else 0
    features['is_summer'] = 1 if season == 'summer' else 0
    features['is_autumn'] = 1 if season == 'autumn' else 0

    # Temperature categories
    features['is_freezing'] = 1 if temperature < 0 else 0
    features['is_cold'] = 1 if 0 <= temperature < 10 else 0
    features['is_mild'] = 1 if 10 <= temperature < 18 else 0
    features['is_warm'] = 1 if 18 <= temperature < 25 else 0
    features['is_hot'] = 1 if temperature >= 25 else 0

    # Calculate cooling degree days (base 18°C)
    features['cooling_degree_days'] = max(0, temperature - 18) / 18
    features['cdd_squared'] = features['cooling_degree_days'] ** 2

    # Weather conditions
    features['has_snow'] = 1 if temperature < 2 and humidity > 80 else 0
    features['is_cloudy'] = 1 if humidity > 70 else 0
    features['cold_and_wet'] = 1 if temperature < 10 and humidity > 70 else 0
    features['hot_and_humid'] = 1 if temperature > 25 and humidity > 70 else 0

    # Wind features
    features['wind_gust_ratio'] = 1.5  # Estimated
    features['wind_lag_1d'] = wind_speed
    features['wind_lag_2d'] = wind_speed
    features['wind_lag_3d'] = wind_speed
    features['wind_lag_7d'] = wind_speed

    # Cyclical time features
    features['month_sin'] = math.sin(2 * math.pi * day_of_year / 365)
    features['day_cos'] = math.cos(2 * math.pi * day_of_year / 365)

    # Squared features
    features['temp_squared'] = temperature ** 2
    features['wind_squared'] = wind_speed ** 2

    # Interaction features
    features['temp_x_is_winter'] = temperature * features['is_winter']
    features['temp_x_is_summer'] = temperature * features['is_summer']
    features['wind_x_temp'] = wind_speed * temperature

    # Solar approximation based on season and cloud cover
    solar_base = 4000
    if season == 'summer':
        solar_base = 6000
    elif season == 'winter':
        solar_base = 2000
    cloud_factor = 1 - (humidity - 50) / 100
    features['shortwave_radiation_sum'] = solar_base * max(0.3, cloud_factor)

    features['solar_x_temp'] = features['shortwave_radiation_sum'] * temperature
    features['solar_x_weekend'] = features['shortwave_radiation_sum'] * features['is_weekend']

    # Year trend (relative to 2002)
    features['year_trend'] = year - 2002

    # Energy lag features (use historical average)
    features['energy_lag_2d'] = TARGET_MEAN
    features['energy_lag_3d'] = TARGET_MEAN
    features['energy_lag_7d'] = TARGET_MEAN
    features['energy_lag_x_weekend'] = TARGET_MEAN * features['is_weekend']

    # Rolling features (use reasonable defaults based on current values)
    features['temp_rolling_std_3d'] = abs(temperature - 10) * 0.2
    features['temp_rolling_std_7d'] = abs(temperature - 10) * 0.3
    features['temp_rolling_std_14d'] = abs(temperature - 10) * 0.35
    features['temp_rolling_14d'] = temperature
    features['precip_rolling_7d'] = humidity * 0.3
    features['precip_rolling_14d'] = humidity * 0.6

    # Wind direction (use average)
    features['wind_direction_10m_mean'] = 220

    # Cloud cover estimation from humidity
    features['cloud_cover_mean'] = min(100, humidity * 0.85)

    return features


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction based on user inputs.
    Returns JSON with prediction and comparison metrics.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Create all features
        features = create_features(data)

        # Build feature vector in correct order
        feature_vector = np.array([[features[col] for col in FEATURE_COLUMNS]])

        # Make prediction
        prediction = model.predict(feature_vector)[0]

        # Calculate metrics
        avg_diff = prediction - TARGET_MEAN
        avg_diff_pct = (avg_diff / TARGET_MEAN) * 100

        # Confidence based on how typical the inputs are
        temp_deviation = abs(data.get('temperature', 10) - 10) / 30
        wind_deviation = abs(data.get('wind_speed', 15) - 15) / 30
        confidence = max(70, min(95, 95 - (temp_deviation + wind_deviation) * 15))

        return jsonify({
            'prediction': round(prediction, 2),
            'prediction_formatted': f"{prediction:,.0f}",
            'avg_consumption': round(TARGET_MEAN, 2),
            'avg_consumption_formatted': f"{TARGET_MEAN:,.0f}",
            'avg_diff': round(avg_diff, 2),
            'avg_diff_pct': round(avg_diff_pct, 1),
            'confidence': round(confidence, 1),
            'model_r2': R2_SCORE,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Return model metadata."""
    return jsonify({
        'model_name': model_info['model_name'],
        'r2_score': R2_SCORE,
        'num_features': len(FEATURE_COLUMNS),
        'target_mean': round(TARGET_MEAN, 2),
        'target_std': round(TARGET_STD, 2),
        'dataset': 'PJM Energy (Pennsylvania) 2002-2018'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
