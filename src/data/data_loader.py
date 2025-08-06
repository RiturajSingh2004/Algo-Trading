import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..utils.logger import setup_logger

def fetch_stock_data(ticker, period="6mo", interval="1d", use_mock=False):
    """
    Fetches stock data from Yahoo Finance or generates mock data.
    """
    logger, _ = setup_logger()
    
    if use_mock:
        return generate_enhanced_mock_data(ticker, period)
    
    try:
        logger.info(f"Fetching real data for {ticker}")
        stock_data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        
        if stock_data.empty:
            logger.warning(f"No data found for {ticker}. Using mock data instead.")
            return generate_enhanced_mock_data(ticker, period)
        
        stock_data.reset_index(inplace=True)
        stock_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        logger.info(f"Successfully fetched {len(stock_data)} rows for {ticker}")
        return stock_data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}. Using mock data.")
        return generate_enhanced_mock_data(ticker, period)

def generate_enhanced_mock_data(ticker, period="6mo"):
    """Generate realistic mock stock data with proper market characteristics."""
    logger, _ = setup_logger()
    logger.info(f"Generating enhanced mock data for {ticker}")
    
    days = 130 if period == "6mo" else 65 if period == "3mo" else 260
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(days * 1.4))
    
    date_rng = pd.date_range(start=start_date, end=end_date, freq='B')[:days]
    
    ticker_params = {
        "RELIANCE.NS": {"base_price": 2400, "volatility": 0.025, "trend": 0.0002},
        "TCS.NS": {"base_price": 3500, "volatility": 0.020, "trend": 0.0003},
        "HDFCBANK.NS": {"base_price": 1600, "volatility": 0.030, "trend": 0.0001},
        "INFY.NS": {"base_price": 1400, "volatility": 0.025, "trend": 0.0002},
        "HINDUNILVR.NS": {"base_price": 2600, "volatility": 0.018, "trend": 0.0001}
    }
    
    params = ticker_params.get(ticker, {"base_price": 1000, "volatility": 0.025, "trend": 0.0001})
    
    np.random.seed(hash(ticker) % 2**32)
    returns = np.random.normal(params["trend"], params["volatility"], len(date_rng))
    
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]
    
    close_prices = params["base_price"] * np.exp(np.cumsum(returns))
    df = pd.DataFrame({'Date': date_rng, 'Close': close_prices})
    
    daily_volatility = params["volatility"] * 0.5
    df['Open'] = df['Close'].shift(1).fillna(df['Close'].iloc[0]) * (1 + np.random.normal(0, daily_volatility/2, len(df)))
    df['High'] = np.maximum(df['Open'], df['Close']) * (1 + np.abs(np.random.normal(0, daily_volatility/3, len(df))))
    df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - np.abs(np.random.normal(0, daily_volatility/3, len(df))))
    df['Volume'] = np.random.lognormal(15, 0.5, len(df)).astype(int)
    
    df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
    df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))
    
    return df.round(2)
