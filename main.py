from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import pickle
from statsmodels.tsa.arima.model import ARIMA
from pydantic import BaseModel
from typing import List, Dict, Optional
from enum import Enum
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cryptocurrency Price Prediction API",
    description="API for predicting cryptocurrency prices using various models",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enums for validation
class ModelType(str, Enum):
    lstm = "lstm"
    nn = "nn"
    gru = "gru"
    arima = "arima"

class CoinType(str, Enum):
    btc = "btc"
    eth = "eth"
    xrp = "xrp"
    bnb = "bnb"
    sol = "sol"

# Model configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    ModelType.lstm: {
        'model': os.path.join(BASE_DIR, 'LSTM_price_prediction.keras'),
        'scaler': os.path.join(BASE_DIR, 'LSTM_price_scaler.pkl')
    },
    ModelType.nn: {
        'model': os.path.join(BASE_DIR, 'NN_price_prediction.keras'),
        'scaler': os.path.join(BASE_DIR, 'NN_price_scaler.pkl')
    },
    ModelType.gru: {
        'model': os.path.join(BASE_DIR, 'gru_model.h5'),
        'scaler': None
    },
    ModelType.arima: {
        'model': os.path.join(BASE_DIR, 'ARIMA_price_prediction.pkl'),
        'scaler': None
    }
}

# Updated Pydantic models for our standard JSON structure
class PricePoint(BaseModel):
    date: str
    actual_price: Optional[float]
    predicted_price: float

def predict_deep_learning(model, data, scaler, prediction_days=60, future_days=30):
    """Helper function for deep learning models (LSTM, NN, GRU)"""
    try:
        logger.debug("Starting deep learning prediction")
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
        
        # Prepare test data
        test_start = dt.datetime(2020,1,1)
        test_end = dt.datetime.now() + dt.timedelta(days=prediction_days)
        test_data = yf.download(f"{data.index.name}-USD", start=test_start, end=test_end)
        actual_prices = test_data['Close'].values
        
        total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
        model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.transform(model_inputs)

        # Prepare test sequences for historical predictions
        x_test = []
        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x-prediction_days:x, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Make historical predictions
        prediction_prices = model.predict(x_test)
        prediction_prices = scaler.inverse_transform(prediction_prices)

        # Calculate future predictions
        future_predictions = []
        last_sequence = model_inputs[-prediction_days:]

        for _ in range(future_days):
            current_sequence = last_sequence.reshape((1, prediction_days, 1))
            next_pred = model.predict(current_sequence, verbose=0)
            next_pred = scaler.inverse_transform(next_pred)[0][0]
            future_predictions.append(float(next_pred))
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = scaler.transform([[next_pred]])[0][0]
            
        return test_data.index, actual_prices, prediction_prices, future_predictions

    except Exception as e:
        logger.error(f"Error in deep learning prediction: {str(e)}")
        raise

@app.get("/predict/{model_type}/{coin}", response_model=List[PricePoint])
async def predict(model_type: ModelType, coin: CoinType):
    """Predict cryptocurrency prices"""
    try:
        logger.info(f"Starting prediction for {coin.value} using {model_type.value} model")
        
        # Load cryptocurrency data
        start = dt.datetime(2016,1,1)
        end = dt.datetime.now()
        ticker = f"{coin.value.upper()}-USD"
        data = yf.download(ticker, start=start, end=end)
        
        if data.empty:
            raise HTTPException(status_code=400, detail=f"No data available for {ticker}")
            
        data.index.name = coin.value.upper()

        result_array = []

        # Load model and make predictions
        if model_type == ModelType.arima:
            with open(MODEL_PATHS[model_type]['model'], 'rb') as file:
                loaded_model = pickle.load(file)
            
            # Prepare test data for ARIMA
            test_start = dt.datetime(2020,1,1)
            test_end = dt.datetime.now()
            test_data = yf.download(ticker, start=test_start, end=test_end)
            
            model = ARIMA(data['Close'], order=loaded_model.order)
            fitted_model = model.fit()
            
            # Historical predictions
            prediction_prices = fitted_model.predict(start=len(data)-len(test_data), end=len(data)-1)
            future_predictions = fitted_model.forecast(steps=30).tolist()
            
            # Create response data
            for date, actual, predicted in zip(test_data.index, test_data['Close'], prediction_prices):
                result_array.append(PricePoint(
                    date=date.strftime('%Y-%m-%d'),
                    actual_price=float(actual),
                    predicted_price=float(predicted)
                ))
                
            # Add future predictions
            future_dates = pd.date_range(start=test_data.index[-1], periods=31)[1:]
            for date, predicted in zip(future_dates, future_predictions):
                result_array.append(PricePoint(
                    date=date.strftime('%Y-%m-%d'),
                    actual_price=None,
                    predicted_price=float(predicted)
                ))
                
        else:
            model = load_model(MODEL_PATHS[model_type]['model'])
            scaler = MinMaxScaler(feature_range=(0,1))
            
            # Get predictions from deep learning model
            historical_dates, actual_prices, prediction_prices, future_predictions = predict_deep_learning(
                model, data, scaler
            )
            
            # Create response data for historical predictions
            for date, actual, predicted in zip(historical_dates, actual_prices, prediction_prices.flatten()):
                result_array.append(PricePoint(
                    date=date.strftime('%Y-%m-%d'),
                    actual_price=float(actual),
                    predicted_price=float(predicted)
                ))
                
            # Add future predictions
            future_dates = pd.date_range(start=historical_dates[-1], periods=31)[1:]
            for date, predicted in zip(future_dates, future_predictions):
                result_array.append(PricePoint(
                    date=date.strftime('%Y-%m-%d'),
                    actual_price=None,
                    predicted_price=float(predicted)
                ))
        
        logger.info(f"Successfully generated predictions for {coin.value}")
        return result_array
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint that also verifies model files"""
    try:
        # Check if all model files exist
        for model_type, paths in MODEL_PATHS.items():
            if not os.path.exists(paths['model']):
                return {
                    "status": "unhealthy",
                    "error": f"Missing model file for {model_type}: {paths['model']}"
                }
            if paths['scaler'] and not os.path.exists(paths['scaler']):
                return {
                    "status": "unhealthy",
                    "error": f"Missing scaler file for {model_type}: {paths['scaler']}"
                }
        
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)