import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Indian Stock Market - RRG Dashboard", layout="wide")
st.title("📊 Sector Rotation - Relative Rotation Graph (RRG)")

def calculate_rrg(tickers, benchmark, interval, window=14, tail_length=5):
    # Determine lookback based on interval to ensure enough history for calculation
    if interval == '60m':
        period = '1mo'
    elif interval == '1d':
        period = '6mo'
    elif interval == '1wk':
        period = '2y'
    else:
        period = 'max'

    all_tickers = tickers + [benchmark]
    
    # Using 'period' instead of hardcoded dates solves holiday/weekend data missing errors
    data = yf.download(all_tickers, period=period, interval=interval, progress=False)
    
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    if 'Close' in data.columns:
        data = data['Close']
        
    data = data.ffill().bfill()
    
    valid_tickers = [t for t in tickers if t in data.columns and not data[t].isna().all()]
    if not valid_tickers or benchmark not in data.columns:
        return pd.DataFrame(), pd.DataFrame()
    
    rs_ratios = pd.DataFrame()
    for t in valid_tickers:
        rs_ratios[t] = (data[t] / data[benchmark]) * 100
        
    if rs_ratios.shape[0] < (window * 2):
        return pd.DataFrame(), pd.DataFrame()
        
    rs_ratio_smoothed = rs_ratios.ewm(span=window, adjust=False).mean()
    mean_rs = rs_ratio_smoothed.rolling(window=window).mean()
    std_rs = rs_ratio_smoothed.rolling(window=window).std()
    jdk_rs_ratio = 100 + ((rs_ratio_smoothed - mean_rs) / std_rs) * 10
    
    rs_momentum = rs_ratio_smoothed.pct_change(periods=window) * 100
    rs_mom_smoothed = rs_momentum.ewm(span=window, adjust=False).mean()
    mean_mom = rs_mom_smoothed.rolling(window=window).mean()
    std_mom = rs_mom_smoothed.rolling(window=window).std()
    jdk_rs_momentum = 100 + ((rs_mom_smoothed - mean_mom) / std_mom) * 10
    
    jdk_rs_ratio = jdk_rs_ratio.dropna().tail(tail_length)
    jdk_rs_momentum = jdk_rs_momentum.dropna().tail(tail_length)
    
    return jdk_rs_ratio, jdk_rs_momentum

def plot_rrg(jdk_rs_ratio, jdk_rs_momentum, timeframe_title):
    if jdk_rs_ratio.empty or jdk_rs_momentum.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data found for this specific timeframe right now.", showarrow=False, font=dict(size=16))
        fig.update_layout(title=f"Timeframe: {timeframe_title}", height=400)
        return fig
        
    fig = go.Figure()
    max_val = max(102, max(jdk_rs_ratio.max().max(), jdk_rs_momentum.max().max()))
    min_val = min(98, min(jdk_rs_ratio.min().min(), jdk_rs_momentum.min().min()))
    padding = max(abs(max_val - 100), abs(100 - min_val)) + 0.5
    
    x_range = [100 - padding, 100 + padding]
    y_range = [100 - padding, 100 + padding]
    
    fig.add_shape(type="rect", x0=100, y0=100, x1=100+padding, y1=100+padding, fillcolor="rgba(0,255,0,0.05)", line_width=0)
    fig.add_shape(type="rect", x0=100, y0=100-padding, x1=100+padding, y1=100, fillcolor="rgba(255,255,0,0.05)", line_width=0)
    fig.add_shape(type="rect", x0=100-padding, y0=100-padding, x1=100, y1=100, fillcolor="rgba(255,0,0,0.05)", line_width=0)
    fig.add_shape(type="rect", x0=100-padding, y0=100, x1=100, y1=100+padding, fillcolor="rgba(0,0,255,0.05)", line_width=0)
    
    for col in jdk_rs_ratio.columns:
        x_data = jdk_rs_ratio[col].values
        y_data = jdk_rs_momentum[col].values
        
        display_name = col.replace('.NS', '')
        
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data, mode='lines+markers',
            name=display_name, line=dict(width=2),
            marker=dict(size=[6]*(len(x_data)-1) + [12], symbol=['circle']*(len(x_data)-1) + ['arrow-bar-up'])
        ))
        fig.add_annotation(
            x=x_data[-1], y=y_data[-1], text=display_name,
            showarrow=True, arrowhead=1, ax=20, ay=-20
        )
        
    fig.add_shape(type="line", x0=100, y0=100-padding, x1=100, y1=100+padding, line=dict(color="black", width=1, dash="dash"))
    fig.add_shape(type="line", x0=100-padding, y0=100, x1=100+padding, y1=100, line=dict(color="black", width=1, dash="dash"))
    
    fig.add_annotation(x=100+padding/2, y=100+padding/1.1, text="LEADING", font=dict(color="green", size=14), showarrow=False)
    fig.add_annotation(x=100+padding/2, y=100-padding/1.1, text="WEAKENING", font=dict(color="orange", size=14), showarrow=False)
    fig.add_annotation(x=100-padding/2, y=100-padding/1.1, text="LAGGING", font=dict(color="red", size=14), showarrow=False)
    fig.add_annotation(x=100-padding/2, y=100+padding/1.1, text="IMPROVING", font=dict(color="blue", size=14), showarrow=False)
    
    fig.update_layout(
        title=f"Timeframe: {timeframe_title}",
        xaxis_title="RS Ratio", yaxis_title="RS Momentum",
        xaxis=dict(range=x_range), yaxis=dict(range=y_range),
        height=600, showlegend=True
    )
    return fig

sectors = ['NIFTYIT.NS', 'NIFTYBANK.NS', 'NIFTYFMCG.NS', 'NIFTYAUTO.NS', 'NIFTYINFRA.NS', 'NIFTYPHARMA.NS', 'NIFTYREALTY.NS', 'NIFTYMETAL.NS']
benchmark_idx = '^NSEI'

st.sidebar.header("Settings")
tail = st.sidebar.slider("Tail Length (History)", min_value=3, max_value=10, value=5)

if st.sidebar.button("🔄 Refresh Data"):
    st.clear_cache()
    st.rerun()

tab1, tab2, tab3, tab4 = st.tabs(["Hourly", "Daily", "Weekly", "Monthly"])

with tab1:
    with st.spinner("Fetching Hourly Data..."):
        rh, mh = calculate_rrg(sectors, benchmark_idx, '60m', tail_length=tail)
        st.plotly_chart(plot_rrg(rh, mh, "Hourly"), use_container_width=True)

with tab2:
    with st.spinner("Fetching Daily Data..."):
        rd, md = calculate_rrg(sectors, benchmark_idx, '1d', tail_length=tail)
        st.plotly_chart(plot_rrg(rd, md, "Daily"), use_container_width=True)

with tab3:
    with st.spinner("Fetching Weekly Data..."):
        rw, mw = calculate_rrg(sectors, benchmark_idx, '1wk', tail_length=tail)
        st.plotly_chart(plot_rrg(rw, mw, "Weekly"), use_container_width=True)

with tab4:
    with st.spinner("Fetching Monthly Data..."):
        rm, mm = calculate_rrg(sectors, benchmark_idx, '1mo', tail_length=tail)
        st.plotly_chart(plot_rrg(rm, mm, "Monthly"), use_container_width=True)

        st.plotly_chart(plot_rrg(rm, mm, "Monthly"), use_container_width=True)
