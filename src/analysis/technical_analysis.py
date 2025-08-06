import pandas as pd
import pandas_ta as ta
from ..utils.logger import setup_logger
from ..config import (
    RSI_PERIOD, SMA_SHORT_PERIOD, SMA_LONG_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, BBANDS_PERIOD, BBANDS_STD
)

def calculate_advanced_indicators(df):
    """Calculate comprehensive technical indicators."""
    logger, _ = setup_logger()
    logger.info("Calculating advanced technical indicators")
    
    if df is None or df.empty:
        logger.error("No data provided for indicator calculation")
        return None
    
    try:
        # Basic indicators
        df.ta.rsi(length=RSI_PERIOD, append=True)
        df.ta.sma(length=SMA_SHORT_PERIOD, append=True)
        df.ta.sma(length=SMA_LONG_PERIOD, append=True)
        df.ta.ema(length=MACD_FAST, append=True)
        df.ta.ema(length=MACD_SLOW, append=True)
        df.ta.macd(append=True)
        
        # Additional indicators
        df.ta.bbands(length=BBANDS_PERIOD, std=BBANDS_STD, append=True)
        df.ta.stoch(append=True)
        df.ta.willr(append=True)
        df.ta.atr(length=RSI_PERIOD, append=True)
        df.ta.ad(append=True)
        df.ta.obv(append=True)
        
        # Custom features
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open']
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Distance_to_Support'] = (df['Close'] - df['Support']) / df['Close']
        df['Distance_to_Resistance'] = (df['Resistance'] - df['Close']) / df['Close']
        
        df.dropna(inplace=True)
        logger.info(f"Successfully calculated indicators. DataFrame shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None
