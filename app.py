from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and preprocess data, then train the model
def preprocess_and_train(data_path):
    data = pd.read_csv(data_path)
    
    # Convert categorical columns to lowercase
    data['day_of_week'] = data['day_of_week'].str.lower()
    data['location'] = data['location'].str.lower()
    data['time_of_day'] = data['time_of_day'].str.lower()
    
    X = data.drop('traffic_volume', axis=1)
    y = data['traffic_volume']
    
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    return pipeline

def predict_traffic_volume(model, temperature, day_of_week, location, time_of_day):
    # Convert input data to lowercase
    day_of_week = day_of_week.lower()
    location = location.lower()
    time_of_day = time_of_day.lower()
    
    new_data = pd.DataFrame({
        'temperature': [temperature],
        'day_of_week': [day_of_week],
        'location': [location],
        'time_of_day': [time_of_day]
    })
    return model.predict(new_data)[0]

data_path = 'J:/Desktop/New folder/traffic-volume-prediction/data/traffic_data.csv'  # Update this with the correct path
model_pipeline = preprocess_and_train(data_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        location = request.form['location']
        day_of_week = request.form['day_of_week']
        time_of_day = request.form['time_of_day']
        
        predicted_traffic_volume = predict_traffic_volume(
            model_pipeline, temperature, day_of_week, location, time_of_day)
        
        return render_template('index.html', prediction_result=predicted_traffic_volume, 
                               temperature=temperature, location=location,
                               day_of_week=day_of_week, time_of_day=time_of_day)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
