import streamlit as st
import pandas as pd
from datetime import datetime
import json
import logging

# Import from our modular structure
from src.config import (
    NIFTY_50_TICKERS, DEFAULT_TICKERS,
    RSI_OVERSOLD, STOCH_OVERSOLD
)
from src.data.data_loader import fetch_stock_data
from src.analysis.technical_analysis import calculate_advanced_indicators
from src.analysis.strategy import apply_enhanced_trading_strategy
from src.models.ml_models import train_ensemble_models
from src.visualization.charts import create_advanced_charts
from src.utils.logger import setup_logger, StreamlitLogHandler

def initialize_session_state(initial_capital):
    """Initialize or reset session state variables."""
    if 'results_generated' not in st.session_state:
        st.session_state.results_generated = False
        st.session_state.all_trade_logs = pd.DataFrame()
        st.session_state.all_summaries = []
        st.session_state.ml_results = {}
        st.session_state.portfolio_summary = {
            'total_capital': initial_capital,
            'total_return': 0,
            'total_return_pct': 0,
            'total_trades': 0,
            'overall_win_rate': 0
        }

def process_stock(ticker, use_mock_data, logger):
    """Process individual stock data and generate analysis."""
    try:
        logger.info(f"Processing {ticker}")
        
        # Fetch and validate data
        stock_data = fetch_stock_data(ticker, use_mock=use_mock_data)
        if stock_data is None or stock_data.empty:
            logger.error(f"No data available for {ticker}")
            return None, None, None
        
        # Calculate technical indicators
        data_with_indicators = calculate_advanced_indicators(stock_data)
        if data_with_indicators is None:
            logger.error(f"Failed to calculate indicators for {ticker}")
            return None, None, None
        
        # Generate trading signals
        signals = apply_enhanced_trading_strategy(data_with_indicators)
        
        # Train ML models
        ml_results = train_ensemble_models(data_with_indicators)
        
        # Create visualization
        chart = create_advanced_charts(data_with_indicators, signals, ticker)
        
        return data_with_indicators, signals, ml_results, chart
        
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return None, None, None, None

def run_trading_algorithm(selected_tickers, use_mock_data, initial_capital):
    """Main algorithm that processes all selected stocks."""
    logger, log_handler = setup_logger()
    log_handler.clear_logs()
    
    # Initialize session state
    initialize_session_state(initial_capital)
    
    # Setup progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful_stocks = 0
    
    # Process each stock
    for i, ticker in enumerate(selected_tickers):
        # Update progress
        progress = (i + 1) / len(selected_tickers)
        progress_bar.progress(progress)
        status_text.text(f"Processing {ticker}... ({i+1}/{len(selected_tickers)})")
        
        # Process stock data
        results = process_stock(ticker, use_mock_data, logger)
        if results is not None:
            data, signals, ml_results, chart = results
            
            # Store ML results if available
            if ml_results is not None:
                model, accuracy, details = ml_results
                st.session_state.ml_results[ticker] = {
                    'accuracy': accuracy,
                    'model_type': type(model).__name__,
                    'details': details
                }
            
            # Display chart if available
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
            
            successful_stocks += 1
    
    # Update completion status
    st.session_state.results_generated = True
    progress_bar.progress(1.0)
    status_text.text("✅ Processing completed!")
    logger.info(f"Algorithm completed. Processed {successful_stocks}/{len(selected_tickers)} stocks successfully")

def create_sidebar():
    """Create and handle sidebar components."""
    st.sidebar.header("🔧 Configuration")
    
    # Data source selection
    use_mock_data = st.sidebar.radio(
        "Data Source",
        ["Real Market Data (Yahoo Finance)", "Mock Data (Demo)"],
        help="Choose between real market data or simulated data for demonstration"
    ) == "Mock Data (Demo)"
    
    # Stock selection
    st.sidebar.subheader("📊 Stock Selection")
    
    # Preset options
    preset_option = st.sidebar.selectbox(
        "Choose Preset",
        ["Custom Selection", "Top 5 NIFTY 50", "Technology Stocks", "Banking Stocks"],
        help="Select a preset group of stocks or choose custom"
    )
    
    # Handle preset selections
    if preset_option == "Custom Selection":
        selected_tickers = st.sidebar.multiselect(
            "Select Stocks",
            NIFTY_50_TICKERS,
            default=DEFAULT_TICKERS[:3],
            help="Choose stocks for analysis"
        )
    else:
        if preset_option == "Top 5 NIFTY 50":
            selected_tickers = DEFAULT_TICKERS
        elif preset_option == "Technology Stocks":
            selected_tickers = ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"]
        else:  # Banking Stocks
            selected_tickers = ["HDFCBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "ICICIBANK.NS", "SBIN.NS"]
    
    # Trading parameters
    st.sidebar.subheader("💰 Trading Parameters")
    initial_capital = st.sidebar.number_input(
        "Initial Capital (₹)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000,
        help="Starting capital for backtesting"
    )
    
    return selected_tickers, use_mock_data, initial_capital

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Advanced Algo-Trading System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Advanced Algorithmic Trading System")
    
    # Create sidebar and get user inputs
    selected_tickers, use_mock_data, initial_capital = create_sidebar()
    
    # Main execution button
    if st.sidebar.button("🚀 Run Trading Algorithm", type="primary"):
        if not selected_tickers:
            st.sidebar.error("Please select at least one stock")
        else:
            with st.spinner("Running trading analysis..."):
                run_trading_algorithm(selected_tickers, use_mock_data, initial_capital)
    
    # Clear results button
    if st.sidebar.button("🗑️ Clear Results"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Display welcome message or results
    if not st.session_state.get('results_generated', False):
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
        <h3>👋 Welcome to the Advanced Algo-Trading System</h3>
        <p>Configure your settings in the sidebar and click <strong>"Run Trading Algorithm"</strong> 
        to begin comprehensive market analysis.</p>
        <p>This system will analyze your selected stocks using advanced technical indicators, 
        machine learning models, and risk management techniques.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
