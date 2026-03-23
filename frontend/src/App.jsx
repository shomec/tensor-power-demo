import React, { useState } from 'react';
import axios from 'axios';
import { Image as ImageIcon, MessageSquare, LineChart, Home, Upload, Send, Play } from 'lucide-react';

const Card = ({ title, icon: Icon, children }) => (
  <div className="card">
    <h2><Icon className="icon" size={24} /> {title}</h2>
    {children}
  </div>
);

const App = () => {
  // MobileNetV2 State
  const [imageFile, setImageFile] = useState(null);
  const [imageResult, setImageResult] = useState(null);
  const [loadingImage, setLoadingImage] = useState(false);

  // Sentiment State
  const [sentimentText, setSentimentText] = useState('');
  const [sentimentResult, setSentimentResult] = useState(null);
  const [loadingSentiment, setLoadingSentiment] = useState(false);

  // Weather State
  const [weatherData, setWeatherData] = useState('72, 75, 74, 78, 80, 79, 82');
  const [weatherResult, setWeatherResult] = useState(null);
  const [loadingWeather, setLoadingWeather] = useState(false);

  // Price Prediction State
  const [area, setArea] = useState(1200);
  const [bedrooms, setBedrooms] = useState(3);
  const [priceResult, setPriceResult] = useState(null);
  const [loadingPrice, setLoadingPrice] = useState(false);

  const handleClassify = async () => {
    if (!imageFile) return;
    setLoadingImage(true);
    const formData = new FormData();
    formData.append('file', imageFile);
    try {
      const res = await axios.post('/api/classify-image', formData);
      setImageResult(res.data.predictions);
    } catch (e) { console.error(e); }
    setLoadingImage(false);
  };

  const handleSentiment = async () => {
    setLoadingSentiment(true);
    try {
      const res = await axios.post('/api/analyze-sentiment', { text: sentimentText });
      setSentimentResult(res.data);
    } catch (e) { console.error(e); }
    setLoadingSentiment(false);
  };

  const handleWeather = async () => {
    setLoadingWeather(true);
    const dataArray = weatherData.split(',').map(n => parseFloat(n.trim())).filter(n => !isNaN(n));
    try {
      const res = await axios.post('/api/forecast-weather', { data: dataArray, steps: 7 });
      setWeatherResult(res.data.forecast);
    } catch (e) { console.error(e); }
    setLoadingWeather(false);
  };

  const handlePrice = async () => {
    setLoadingPrice(true);
    try {
      const res = await axios.post('/api/predict-price', { area, bedrooms });
      setPriceResult(res.data.predicted_price);
    } catch (e) { console.error(e); }
    setLoadingPrice(false);
  };

  return (
    <div className="container">
      <header>
        <h1>Tensor Power Demo UI</h1>
        <p>Showcasing the power of Machine Learning with 4 dynamic use cases</p>
      </header>

      <div className="grid">
        {/* Image Classification */}
        <Card title="Image Recognition" icon={ImageIcon}>
          <div className="input-group">
            <label>Upload an image to classify</label>
            <input type="file" onChange={(e) => setImageFile(e.target.files[0])} accept="image/*" />
          </div>
          <button onClick={handleClassify} disabled={!imageFile || loadingImage}>
            {loadingImage ? <div className="loading-spinner" /> : <><Upload size={18} /> Classify Image</>}
          </button>
          {imageResult && (
            <div className="result">
              {imageResult.map((p, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span>{p.description}</span>
                  <span style={{ color: 'var(--primary)' }}>{(p.probability * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Sentiment Analysis */}
        <Card title="Sentiment Analysis" icon={MessageSquare}>
          <div className="input-group">
            <label>Enter text to analyze sentiment</label>
            <textarea
              rows="3"
              placeholder="E.g. TensorFlow is absolutely incredible!"
              value={sentimentText}
              onChange={(e) => setSentimentText(e.target.value)}
            />
          </div>
          <button onClick={handleSentiment} disabled={!sentimentText || loadingSentiment}>
            {loadingSentiment ? <div className="loading-spinner" /> : <><Send size={18} /> Analyze Tone</>}
          </button>
          {sentimentResult && (
            <div className="result" style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: sentimentResult.sentiment === 'Neutral' ? '#94a3b8' : sentimentResult.sentiment === 'Positive' ? '#4ade80' : '#f87171' }}>
                {sentimentResult.sentiment}
              </div>
              <div style={{ color: 'var(--text-muted)' }}>Confidence: {(sentimentResult.confidence * 100).toFixed(1)}%</div>
            </div>
          )}
        </Card>

        {/* Weather Forecasting */}
        <Card title="Weather Forecast" icon={LineChart}>
          <div className="input-group">
            <label>Last 7 days temp (&deg;F, comma-separated)</label>
            <input
              value={weatherData}
              onChange={(e) => setWeatherData(e.target.value)}
            />
          </div>
          <button onClick={handleWeather} disabled={loadingWeather}>
            {loadingWeather ? <div className="loading-spinner" /> : <><Play size={18} /> Predict Next 7 Days</>}
          </button>
          {weatherResult && (
            <div className="result">
              <div style={{ marginBottom: '8px', fontWeight: 'bold' }}>Next 7 days (&deg;F):</div>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {weatherResult.map((v, i) => (
                  <span key={i} style={{ background: 'var(--primary)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.8rem' }}>{v}</span>
                ))}
              </div>
            </div>
          )}
        </Card>

        {/* House Price Prediction */}
        <Card title="Real Estate Prediction" icon={Home}>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <div className="input-group" style={{ flex: 1 }}>
              <label>Area (sqft)</label>
              <input type="number" value={area} onChange={(e) => setArea(e.target.value)} />
            </div>
            <div className="input-group" style={{ flex: 1 }}>
              <label>Bedrooms</label>
              <input type="number" value={bedrooms} onChange={(e) => setBedrooms(e.target.value)} />
            </div>
          </div>
          <button onClick={handlePrice} disabled={loadingPrice}>
            {loadingPrice ? <div className="loading-spinner" /> : "Predict Price"}
          </button>
          {priceResult !== null && (
            <div className="result" style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--primary)' }}>
                ${priceResult.toLocaleString()}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Estimated Market Value</div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};

export default App;
