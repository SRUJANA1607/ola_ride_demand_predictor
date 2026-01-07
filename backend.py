from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Create FastAPI app
app = FastAPI(title="Ola Ride Demand Predictor API", version="1.0.0")

# Mock ML model (dummy regression)
class DummyRegressor:
    def predict(self, X):
        # Generate random predictions based on input features
        predictions = []
        for row in X:
            # Simple logic: higher traffic + surge = higher demand
            base_demand = 50 + random.randint(0, 100)
            traffic_factor = row[4] * 10  # traffic level
            surge_factor = (row[5] - 1) * 20  # surge multiplier
            time_factor = 20 if 7 <= row[1] <= 22 else -10  # peak hours
            weather_factor = -15 if row[2] == 1 else 0  # rainy weather reduces demand

            demand = base_demand + traffic_factor + surge_factor + time_factor + weather_factor
            demand = max(10, min(300, demand))  # clamp between 10-300
            predictions.append(demand)
        return np.array(predictions)

# Load dummy model
model = DummyRegressor()

# Pydantic models
class PredictionRequest(BaseModel):
    location: str
    time_of_day: float
    day_of_week: int
    weather: str
    traffic_level: float
    surge_multiplier: float

class PredictionResponse(BaseModel):
    predicted_demand: float
    demand_level: str
    confidence_score: float

class AnalyticsResponse(BaseModel):
    hourly_average: List[Dict[str, Any]]
    weather_impact: Dict[str, float]
    peak_hours: List[int]

class TrendsResponse(BaseModel):
    daily_trends: List[Dict[str, Any]]
    insights: str

# Weather mapping
weather_map = {'Sunny': 0, 'Rainy': 1, 'Cloudy': 2}

# Location zones
zones = [
    "Hyderabad North", "Gachibowli", "Hitech City", "Banjara Hills",
    "Jubilee Hills", "Secunderabad", "Kukatpally", "Ameerpet"
]

@app.post("/predict", response_model=PredictionResponse)
async def predict_demand(request: PredictionRequest):
    try:
        # Validate inputs
        if request.location not in zones:
            raise HTTPException(status_code=400, detail="Invalid location")
        if not (0 <= request.time_of_day <= 24):
            raise HTTPException(status_code=400, detail="Time of day must be between 0-24")
        if not (0 <= request.day_of_week <= 6):
            raise HTTPException(status_code=400, detail="Day of week must be between 0-6")
        if request.weather not in weather_map:
            raise HTTPException(status_code=400, detail="Invalid weather")
        if not (1 <= request.traffic_level <= 10):
            raise HTTPException(status_code=400, detail="Traffic level must be between 1-10")
        if not (1.0 <= request.surge_multiplier <= 3.0):
            raise HTTPException(status_code=400, detail="Surge multiplier must be between 1.0-3.0")

        # Prepare features for model
        features = np.array([[
            zones.index(request.location),  # location index
            request.time_of_day,
            weather_map[request.weather],
            request.day_of_week,
            request.traffic_level,
            request.surge_multiplier
        ]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Determine demand level
        if prediction >= 200:
            demand_level = "High"
        elif prediction >= 100:
            demand_level = "Medium"
        else:
            demand_level = "Low"

        # Mock confidence score
        confidence = round(0.7 + random.random() * 0.25, 2)

        return PredictionResponse(
            predicted_demand=round(prediction, 1),
            demand_level=demand_level,
            confidence_score=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    # Mock hourly averages
    hourly_average = []
    for hour in range(24):
        base_demand = 80 + (hour - 12) ** 2 * -2  # peak around noon
        demand = max(20, base_demand + random.randint(-20, 20))
        hourly_average.append({
            "hour": hour,
            "average_demand": round(demand, 1)
        })

    # Mock weather impact
    weather_impact = {
        "Sunny": 1.2,
        "Cloudy": 0.9,
        "Rainy": 0.6
    }

    # Mock peak hours
    peak_hours = [8, 9, 12, 13, 17, 18, 19, 20]

    return AnalyticsResponse(
        hourly_average=hourly_average,
        weather_impact=weather_impact,
        peak_hours=peak_hours
    )

@app.get("/trends", response_model=TrendsResponse)
async def get_trends():
    # Mock daily trends for past 7 days
    daily_trends = []
    base_date = datetime.now() - timedelta(days=7)

    for i in range(7):
        date = base_date + timedelta(days=i)
        demand = 150 + random.randint(-30, 30)
        daily_trends.append({
            "date": date.strftime("%Y-%m-%d"),
            "demand": round(demand, 1),
            "day": date.strftime("%A")
        })

    # Mock insights
    insights = "Peak demand observed on weekdays between 8-10 AM and 5-8 PM. Weather significantly impacts ride requests, with 40% reduction during rainy conditions."

    return TrendsResponse(
        daily_trends=daily_trends,
        insights=insights
    )

@app.get("/")
async def root():
    return {"message": "Ola Ride Demand Predictor API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)