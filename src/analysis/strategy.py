import pandas as pd
from ..utils.logger import setup_logger
from ..config import RSI_OVERSOLD, STOCH_OVERSOLD, MIN_SUPPORT_DISTANCE

def apply_enhanced_trading_strategy(df):
    """
    Enhanced trading strategy with multiple signal confirmations.
    """
    logger, _ = setup_logger()
    
    if df is None or df.empty or 'RSI_14' not in df.columns:
        logger.warning("Insufficient data for strategy application")
        return pd.DataFrame()
    
    logger.info("Applying enhanced trading strategy")
    
    try:
        # Primary conditions
        rsi_oversold = df['RSI_14'] < RSI_OVERSOLD
        sma_crossover = (df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))
        volume_confirm = df['Volume'] > df['Volume'].rolling(20).mean()
        
        # Additional confirmations
        macd_bullish = df['MACD_12_26_9'] > df['MACDs_12_26_9']
        stoch_oversold = df['STOCHk_14_3_3'] < STOCH_OVERSOLD
        price_above_support = df['Distance_to_Support'] > MIN_SUPPORT_DISTANCE
        
        # Combine conditions
        strong_buy = rsi_oversold & sma_crossover & volume_confirm & macd_bullish
        confirmed_buy = strong_buy & stoch_oversold & price_above_support
        
        # Generate signals
        df['Signal'] = 0
        df.loc[strong_buy, 'Signal'] = 1  # Standard buy
        df.loc[confirmed_buy, 'Signal'] = 2  # Strong buy
        
        # Extract buy signals
        buy_signals = df[df['Signal'] > 0].copy()
        buy_signals['Signal_Type'] = buy_signals['Signal'].map({1: 'BUY', 2: 'STRONG_BUY'})
        
        logger.info(f"Generated {len(buy_signals)} buy signals ({len(buy_signals[buy_signals['Signal'] == 2])} strong)")
        
        return buy_signals
        
    except Exception as e:
        logger.error(f"Error applying trading strategy: {e}")
        return pd.DataFrame()
