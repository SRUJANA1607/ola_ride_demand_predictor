import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

def load_or_create_data():
    """
    Load or create the dataset 'ola_rides.csv' with columns: hour, day, temperature, weather, rides.
    Since it's a demo, we'll create synthetic data.
    """
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    hours = np.random.randint(0, 24, n_samples)
    days = np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], n_samples)
    temperatures = np.random.uniform(10, 40, n_samples)
    weathers = np.random.choice(['Sunny', 'Rainy', 'Cloudy'], n_samples)
    # Simulate rides based on hour, day, temp, weather
    rides = (hours * 10 + (temperatures - 20) * 5 + np.where(weathers == 'Sunny', 20, np.where(weathers == 'Cloudy', 10, -10)) +
             np.where(np.isin(days, ['Saturday', 'Sunday']), 30, 0) + np.random.normal(0, 20, n_samples)).astype(int)
    rides = np.maximum(rides, 0)  # Ensure non-negative

    df = pd.DataFrame({
        'hour': hours,
        'day': days,
        'temperature': temperatures,
        'weather': weathers,
        'rides': rides
    })
    df.to_csv('ola_rides.csv', index=False)
    return df

def preprocess_data(df):
    """
    Preprocess the data: encode categorical columns using LabelEncoder.
    """
    le_day = LabelEncoder()
    le_weather = LabelEncoder()
    df['day_encoded'] = le_day.fit_transform(df['day'])
    df['weather_encoded'] = le_weather.fit_transform(df['weather'])
    # Drop original categorical columns
    df = df.drop(['day', 'weather'], axis=1)
    return df, le_day, le_weather

def train_model(X_train, y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using Mean Absolute Error (MAE).
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f}")
    return y_pred

def predict_rides(model, le_day, le_weather, hour, day, temperature, weather):
    """
    Predict number of ride requests for given input.
    """
    day_encoded = le_day.transform([day])[0]
    weather_encoded = le_weather.transform([weather])[0]
    input_data = pd.DataFrame({
        'hour': [hour],
        'temperature': [temperature],
        'day_encoded': [day_encoded],
        'weather_encoded': [weather_encoded]
    })
    prediction = model.predict(input_data)[0]
    return max(0, int(prediction))  # Ensure non-negative

def visualize_predictions(y_test, y_pred):
    """
    Visualize actual vs predicted ride requests using a scatter plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Rides')
    plt.ylabel('Predicted Rides')
    plt.title('Actual vs Predicted Ride Requests')
    plt.grid(True)
    plt.show()

def main():
    # Load or create data
    df = load_or_create_data()

    # Preprocess data
    df_processed, le_day, le_weather = preprocess_data(df)

    # Split data
    X = df_processed.drop('rides', axis=1)
    y = df_processed['rides']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    y_pred = evaluate_model(model, X_test, y_test)

    # Predict for example input
    example_hour = 9
    example_day = 'Monday'
    example_temp = 30
    example_weather = 'Sunny'
    predicted_rides = predict_rides(model, le_day, le_weather, example_hour, example_day, example_temp, example_weather)
    print(f"Predicted number of ride requests for {example_day}, {example_hour} AM, {example_temp}°C, {example_weather}: {predicted_rides}")

    # Visualize
    visualize_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()