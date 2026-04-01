# Tensor Power Demo UI

A machine learning showcase application featuring 4 dynamic use cases.

## Tech Stack

- **Frontend**: React 18, Vite, Lucide React (Icons), Axios
- **Backend**: FastAPI (Python 3.10), Uvicorn
- **Core ML**: **TensorFlow 2.15** (ResNet50V2), **Scikit-learn** (GradientBoosting, California Housing), Statsmodels (ARIMA), HuggingFace Transformers (DistilBERT)
- **Infrastructure**: Docker, Docker Compose, Nginx (Reverse Proxy)

## ML Use Cases

### 1. 🖼️ Image Recognition — Computer Vision
Powered by **ResNet50V2** (pre-trained on ImageNet). Upload any photo and the model identifies it across 1,000 categories with a confidence score.

### 2. 💬 Sentiment Analysis — NLP
Uses **`lxyuan/distilbert-base-multilingual-cased-sentiments-student`** (TensorFlow / HuggingFace). Returns Positive / Neutral / Negative with a confidence score. Supports multilingual input.

### 3. 🌤️ Weather Forecast — Time-Series
**ARIMA(1,0,0)** model (Statsmodels). Provide the last 7 days of temperature readings and get a 7-day forecast.

### 4. 🏠 Real Estate Price Prediction — Regression
**Real ML model** — `GradientBoostingRegressor` (200 trees, max_depth=4, subsample=0.8) trained at startup on the **California Housing dataset** (~20,000 real census records).

| Input | Description |
|---|---|
| `area` (sqft) | House square footage — mapped to average room count |
| `bedrooms` | Number of bedrooms |
| `age_years` _(optional)_ | Age of the property (default: 20) |
| `latitude` / `longitude` _(optional)_ | Location (default: San Francisco) |

**Response includes:**
- `predicted_price` — model's dollar estimate
- `confidence_interval` — ±15% low/high range
- `r2_score` — model accuracy on held-out test set (~0.83)
- `model` — model name for transparency

**Example:**
```json
POST /predict-price
{ "area": 1500, "bedrooms": 3, "age_years": 15 }

{
  "predicted_price": 387420.50,
  "confidence_interval": { "low": 329307.43, "high": 445533.58 },
  "model": "GradientBoostingRegressor (California Housing)",
  "r2_score": 0.8314
}
```

---

## Prerequisites

- [Docker](https://www.docker.com/get-started) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed

## Setup & Running

```bash
# 1. Navigate to the project directory
cd tensor-power-demo

# 2. Build and start all containers
docker compose up --build
```

- **Frontend:** http://localhost
- **Backend API Docs (Swagger):** http://localhost:8000/docs

---

![Result Screenshot](./result_screenshot.png)
