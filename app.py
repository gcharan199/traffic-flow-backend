from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and train the model
def preprocess_and_train(csv_path):
    df = pd.read_csv(csv_path)

    # Normalize string inputs
    df['day_of_week'] = df['day_of_week'].str.lower()
    df['location'] = df['location'].str.lower()
    df['time_of_day'] = df['time_of_day'].str.lower()

    X = df.drop('traffic_volume', axis=1)
    y = df['traffic_volume']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['day_of_week', 'location', 'time_of_day'])
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    return pipeline

# Path to your dataset (same folder)
model = preprocess_and_train('traffic_data.csv')

# Route for browser (form interface)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        temp = float(request.form['temperature'])
        day = request.form['day_of_week']
        location = request.form['location']
        time = request.form['time_of_day']

        input_df = pd.DataFrame([{
            'temperature': temp,
            'day_of_week': day.lower(),
            'location': location.lower(),
            'time_of_day': time.lower()
        }])

        prediction = model.predict(input_df)[0]
        return render_template('index.html', prediction=round(prediction, 2))

    return render_template('index.html')

# ✅ API endpoint for React Native
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([{
            'temperature': float(data['temperature']),
            'day_of_week': data['day_of_week'].lower(),
            'location': data['location'].lower(),
            'time_of_day': data['time_of_day'].lower()
        }])

        prediction = model.predict(input_df)[0]
        return jsonify({'traffic_volume': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
