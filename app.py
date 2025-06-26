from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# Load and train the model
def preprocess_and_train(csv_path):
    df = pd.read_csv(csv_path)

    df['day_of_week'] = df['day_of_week'].str.lower()
    df['location'] = df['location'].str.lower()
    df['time_of_day'] = df['time_of_day'].str.lower()

    X = df.drop('traffic_volume', axis=1)
    y = df['traffic_volume']

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), ['day_of_week', 'location', 'time_of_day'])],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    pipeline.fit(X, y)
    return pipeline

model = preprocess_and_train('traffic_data.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.content_type != 'application/json':
        return jsonify({'error': 'Content-Type must be application/json'}), 415

    try:
        data = request.get_json()
        input_df = pd.DataFrame([{
            'temperature': float(data['temperature']),
            'day_of_week': data['day_of_week'].lower(),
            'location': data['location'].lower(),
            'time_of_day': data['time_of_day'].lower()
        }])
        prediction = model.predict(input_df)[0]
        return jsonify({'traffic_volume': int(round(prediction))})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

import os

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)

