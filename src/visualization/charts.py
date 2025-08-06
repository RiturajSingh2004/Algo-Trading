import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_advanced_charts(df, signals, ticker):
    """Create comprehensive trading charts."""
    if df is None or df.empty:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'{ticker} - Price & Moving Averages',
            'RSI',
            'MACD',
            'Volume'
        ),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Price chart with moving averages
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ), row=1, col=1
    )
    
    # Moving averages
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA 50', line=dict(color='blue')),
            row=1, col=1
        )
    
    # Add buy signals
    if not signals.empty:
        fig.add_trace(
            go.Scatter(
                x=signals['Date'],
                y=signals['Close'],
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Buy Signals'
            ), row=1, col=1
        )
    
    # RSI
    if 'RSI_14' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['RSI_14'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=30, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    
    # MACD
    if 'MACD_12_26_9' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MACD_12_26_9'], name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        if 'MACDs_12_26_9' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df['MACDs_12_26_9'], name='Signal', line=dict(color='red')),
                row=3, col=1
            )
        if 'MACDh_12_26_9' in df.columns:
            fig.add_trace(
                go.Bar(x=df['Date'], y=df['MACDh_12_26_9'], name='Histogram'),
                row=3, col=1
            )
    
    # Volume
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='lightblue'),
        row=4, col=1
    )
    
    fig.update_layout(
        title=f"{ticker} - Technical Analysis Dashboard",
        height=800,
        xaxis_rangeslider_visible=False
    )
    
    return fig
