import io
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
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

# Load Sentiment Analysis model globally (Real TF Model with 3 classes: Neg, Neu, Pos)
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", framework="tf")

def real_sentiment_analysis(text: str):
    result = sentiment_pipeline(text)[0]
    # result is like {'label': 'neutral', 'score': 0.99}
    # Labels for this model are: 'negative', 'neutral', 'positive'
    return {
        "sentiment": result['label'].capitalize(),
        "confidence": float(result['score'])
    }

# For the Price Prediction (Regression), we'll use a simple pre-trained weight set or define a small model
# Here we define a simple linear regression model for House Prices (Price = Area * 200 + Bedrooms * 50000)
def predict_house_price(area: float, bedrooms: int):
    # This represents a simple ML model prediction
    price = (area * 300) + (bedrooms * 25000) + 50000
    return {"predicted_price": float(price)}

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
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
    area = float(data.get("area", 0))
    bedrooms = int(data.get("bedrooms", 0))
    return predict_house_price(area, bedrooms)

@app.get("/")
def read_root():
    return {"message": "TensorFlow Demo API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
