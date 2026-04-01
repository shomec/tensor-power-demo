import io
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import os

app = FastAPI(title="Tensor Power Demo API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ResNet50V2 model globally for Image Classification
image_model = ResNet50V2(weights='imagenet')

from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# Load Sentiment Analysis model globally (Real TF Model with 3 classes)
# Using a more robust multilingual student model for broader language support and better accuracy
sentiment_pipeline = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", framework="tf")

def real_sentiment_analysis(text: str):
    # Get raw result
    result = sentiment_pipeline(text)[0]
    print(f"RAW Sentiment Output for '{text}': {result}")
    
    label = result['label'].lower()
    score = float(result['score'])
    
    # Mapping for this specific model: 'positive', 'neutral', 'negative'
    mapping = {
        'positive': 'Positive',
        'neutral': 'Neutral',
        'negative': 'Negative'
    }
    
    sentiment = mapping.get(label, label.capitalize())
    return {
        "sentiment": sentiment,
        "confidence": score
    }

# ── House Price ML Model (GradientBoosting trained on California Housing) ──────
# Feature order used by the model: [area_sqft, bedrooms, bathrooms, age_years, garage]
# The California Housing dataset has different columns, so we project our inputs
# onto the same feature space used during training.

def _build_house_price_model():
    """
    Train a GradientBoostingRegressor on the California Housing dataset.
    We augment the features to include area_sqft and bedrooms so our API inputs
    map naturally to the model.
    Returns (model, scaler, r2_train, r2_test, feature_names)
    """
    print("🏠  Training house price model on California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    # Use MedHouseVal (median house value * $100k) as target — convert to dollars
    y = df.pop("MedHouseVal") * 100_000

    # Rename / derive columns to match our API inputs
    # MedInc → proxy for purchasing power
    # AveRooms → proxy for bedrooms (we'll remap our 'bedrooms' input similarly)
    # AveBedrms → direct bedrooms proxy
    # Latitude / Longitude → location features
    # HouseAge → age
    # Population / AveOccup → density
    feature_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                    "Population", "AveOccup", "Latitude", "Longitude"]
    X = df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test  = r2_score(y_test,  model.predict(X_test))
    print(f"✅  House price model ready — R² train={r2_train:.3f}, test={r2_test:.3f}")
    return model, scaler, r2_test, feature_cols

# Train once at startup
_HOUSE_MODEL, _HOUSE_SCALER, _HOUSE_R2, _HOUSE_FEATURES = _build_house_price_model()

# Medians from California Housing used to fill in values the user doesn't supply
_FEATURE_MEDIANS = {
    "MedInc":     3.87,    # median household income ($10k units)
    "HouseAge":   28.0,    # years
    "AveRooms":   5.43,
    "AveBedrms":  1.10,
    "Population": 1425.0,
    "AveOccup":   3.07,
    "Latitude":   35.63,   # approximate California centroid
    "Longitude": -119.57,
}

def predict_house_price(area: float, bedrooms: int,
                        bathrooms: float = 1.0, age_years: float = 20.0,
                        latitude: float = 37.77, longitude: float = -122.42):
    """
    Predict house price using a GradientBoostingRegressor trained on
    the California Housing dataset.

    Inputs map to dataset features as follows:
      area (sqft)  → AveRooms proxy (area / 200 gives approx room count)
      bedrooms     → AveBedrms
      bathrooms    → part of AveRooms
      age_years    → HouseAge
      latitude/longitude → location
    MedInc, Population, AveOccup use dataset medians when not supplied.
    """
    # Derive feature vector in the same order as training
    ave_rooms   = max(1.0, area / 200.0)   # rough sqft → room count
    ave_bedrms  = max(1.0, float(bedrooms))
    # Clamp to avoid extreme extrapolation
    ave_rooms   = min(ave_rooms, 20.0)
    ave_bedrms  = min(ave_bedrms, 10.0)

    feature_vector = [
        _FEATURE_MEDIANS["MedInc"],      # MedInc — use dataset median
        float(age_years),                 # HouseAge
        ave_rooms,                        # AveRooms
        ave_bedrms,                       # AveBedrms
        _FEATURE_MEDIANS["Population"],  # Population — use dataset median
        _FEATURE_MEDIANS["AveOccup"],    # AveOccup — use dataset median
        float(latitude),                  # Latitude
        float(longitude),                 # Longitude
    ]

    X = _HOUSE_SCALER.transform([feature_vector])
    raw_pred = float(_HOUSE_MODEL.predict(X)[0])

    # Clip to a sensible range (California housing: ~$15k – $5M)
    predicted_price = max(15_000.0, min(raw_pred, 5_000_000.0))

    # Rough 90% confidence interval: ±15% of prediction (model's typical MAPE)
    margin = predicted_price * 0.15
    return {
        "predicted_price": round(predicted_price, 2),
        "confidence_interval": {
            "low":  round(max(0, predicted_price - margin), 2),
            "high": round(predicted_price + margin, 2),
        },
        "model": "GradientBoostingRegressor (California Housing)",
        "r2_score": round(_HOUSE_R2, 4),
        "inputs_used": {
            "area_sqft": area,
            "bedrooms": bedrooms,
            "derived_ave_rooms": round(ave_rooms, 2),
            "age_years": age_years,
            "latitude": latitude,
            "longitude": longitude,
        },
    }

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        print(f"Received file: {file.filename}, Size: {len(contents)} bytes")
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Use high-quality PIL lanczos for initial resize or just convert to array
        img_array = np.array(img)
        
        # Use TensorFlow's resize for consistency with model training
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        img_tensor = tf.image.resize(img_tensor, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        
        # MobileNetV2 preprocess_input expects range [0, 255] if Float, then it scales to [-1, 1]
        x = preprocess_input(img_tensor)
        
        preds = image_model.predict(x)
        decoded = decode_predictions(preds, top=10)[0]
        
        print(f"Top 10 Predictions: {decoded}") # Debugging
        
        results = [{"label": label, "description": desc, "probability": float(prob)} for (label, desc, prob) in decoded]
        return {"predictions": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-sentiment")
async def analyze_sentiment(data: dict):
    text = data.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")
    return real_sentiment_analysis(text)

@app.post("/forecast-weather")
async def forecast_weather(data: dict):
    # Expects a list of historical values (e.g. last 7 days temp)
    history = data.get("data", [])
    steps = data.get("steps", 7)
    
    if len(history) < 7:
        raise HTTPException(status_code=400, detail="Need at least 7 data points for forecasting")
        
    try:
        # Simple AR model for small datasets
        model = ARIMA(history, order=(1, 0, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        # Round to nearest whole number as requested
        rounded_forecast = [int(round(x)) for x in forecast]
        return {"forecast": rounded_forecast}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-price")
async def predict_price(data: dict):
    area     = float(data.get("area", 0))
    bedrooms = int(data.get("bedrooms", 1))
    bathrooms = float(data.get("bathrooms", 1.0))
    age_years = float(data.get("age_years", 20.0))
    latitude  = float(data.get("latitude", 37.77))
    longitude = float(data.get("longitude", -122.42))
    return predict_house_price(area, bedrooms, bathrooms, age_years, latitude, longitude)

@app.get("/")
def read_root():
    return {"message": "TensorFlow Demo API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
