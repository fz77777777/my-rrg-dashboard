import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Premium Responsive Dashboard Configuration
st.set_page_config(
    page_title="NSE India Sectors RRG Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark UI Accent & Styling
st.markdown("""
    <style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    h1 { font-weight: 800; color: #0F172A; letter-spacing: -1px; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; padding: 10px 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("🇮🇳 Indian Stock Market Sector Rotation (NSE RRG Dashboard)")
st.caption("🛡️ 30-Min Intraday Engine Enabled | Benchmark: NIFTY 50 (^NSEI) | Auto-Refresh: 30 Min")

# 💡 30-MINUTE SMART CACHE FOR NSE INTRADAY + HISTORICAL DATA
@st.cache_data(ttl=1800, show_spinner="Fetching NSE 30-Min Intraday Feed...")
def calculate_rrg_cached(tickers_dict, benchmark, interval, window=14, tail_length=5, history_offset=0):
    now = datetime.now()
    
    # Intraday (30m) limits lookback to avoid Yahoo API 400 errors
    if interval == '30m':
        buffer_days = history_offset + 5
        start_date = (now - timedelta(days=25 + buffer_days)).strftime('%Y-%m-%d')
    else:
        buffer_days = history_offset * 7 if interval in ['1wk', '1mo'] else history_offset + 10
        if interval == '1d': start_date = (now - timedelta(days=365 + buffer_days)).strftime('%Y-%m-%d')
        elif interval == '1wk': start_date = (now - timedelta(days=730 + buffer_days)).strftime('%Y-%m-%d')
        else: start_date = (now - timedelta(days=1500 + buffer_days)).strftime('%Y-%m-%d')
        
    end_date = (now + timedelta(days=1)).strftime('%Y-%m-%d')

    all_tickers = list(tickers_dict.keys()) + [benchmark]
    batch_size = 15
    combined_df = pd.DataFrame()
    
    try:
        for i in range(0, len(all_tickers), batch_size):
            batch = all_tickers[i:i+batch_size]
            batch_data = yf.download(
                batch, start=start_date, end=end_date, 
                interval=interval, auto_adjust=True, progress=False
            )
            
            if not batch_data.empty:
                if isinstance(batch_data.columns, pd.MultiIndex):
                    if 'Close' in batch_data.columns.levels[0]:
                        batch_close = batch_data['Close']
                    else:
                        continue
                else:
                    if 'Close' in batch_data.columns:
                        batch_close = batch_data[['Close']]
                    else:
                        batch_close = batch_data
                        
                combined_df = pd.concat([combined_df, batch_close], axis=1)
                
        if combined_df.empty:
            return pd.DataFrame(), pd.DataFrame(), "No Data"
            
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        combined_df = combined_df.ffill().bfill()
        
        valid_tickers = [t for t in tickers_dict.keys() if t in combined_df.columns and not combined_df[t].isna().all()]
        
        if not valid_tickers or benchmark not in combined_df.columns:
            return pd.DataFrame(), pd.DataFrame(), "No Data"
        
        # RRG Mathematical Modeling
        rs_ratios = pd.DataFrame()
        for t in valid_tickers:
            rs_ratios[t] = (combined_df[t] / combined_df[benchmark]) * 100
            
        if rs_ratios.shape[0] < (window * 2):
            return pd.DataFrame(), pd.DataFrame(), "No Data"
            
        rs_ratio_smoothed = rs_ratios.ewm(span=window, adjust=False).mean()
        mean_rs = rs_ratio_smoothed.rolling(window=window).mean()
        std_rs = rs_ratio_smoothed.rolling(window=window).std()
        jdk_rs_ratio = 100 + ((rs_ratio_smoothed - mean_rs) / std_rs) * 10
        
        rs_momentum = rs_ratio_smoothed.pct_change(periods=window) * 100
        rs_mom_smoothed = rs_momentum.ewm(span=window, adjust=False).mean()
        mean_mom = rs_mom_smoothed.rolling(window=window).mean()
        std_mom = rs_mom_smoothed.rolling(window=window).std()
        jdk_rs_momentum = 100 + ((rs_mom_smoothed - mean_mom) / std_mom) * 10
        
        jdk_rs_ratio = jdk_rs_ratio.dropna()
        jdk_rs_momentum = jdk_rs_momentum.dropna()
        
        # 📊 HISTORICAL SNAPSHOT ENGINE
        if history_offset > 0:
            if history_offset >= len(jdk_rs_ratio):
                history_offset = len(jdk_rs_ratio) - tail_length - 1
            
            jdk_rs_ratio = jdk_rs_ratio.iloc[:-history_offset]
            jdk_rs_momentum = jdk_rs_momentum.iloc[:-history_offset]
            
        # Time and Date separation formatting for intraday clarity
        if interval == '30m':
            snapshot_date = jdk_rs_ratio.index[-1].strftime('%b %d, %Y | %H:%M')
        else:
            snapshot_date = jdk_rs_ratio.index[-1].strftime('%B %d, %Y')
        
        return jdk_rs_ratio.tail(tail_length), jdk_rs_momentum.tail(tail_length), snapshot_date
        
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), "Error"

def plot_rrg_labeled(jdk_rs_ratio, jdk_rs_momentum, tickers, title_date):
    if jdk_rs_ratio.empty or jdk_rs_momentum.empty or len(jdk_rs_ratio.columns) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ Data Unavailable for this snapshot window.<br>Please reduce history offset or adjust sector filters.", 
            showarrow=False, font=dict(size=16, color="#64748B")
        )
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor='white', height=400)
        return fig
        
    fig = go.Figure()
    
    all_x = jdk_rs_ratio.values.flatten()
    all_y = jdk_rs_momentum.values.flatten()
    
    if len(all_x) == 0 or len(all_y) == 0:
        max_pad = 5.0
    else:
        max_pad = max(abs(all_x.max() - 100), abs(100 - all_x.min()), abs(all_y.max() - 100), abs(100 - all_y.min())) + 1.5
    
    x_range = [100 - max_pad, 100 + max_pad]
    y_range = [100 - max_pad, 100 + max_pad]
    
    # Quadrant Shading
    fig.add_shape(type="rect", x0=100, y0=100, x1=100+max_pad, y1=100+max_pad, fillcolor="rgba(34,197,94,0.02)", line_width=0)
    fig.add_shape(type="rect", x0=100, y0=100-max_pad, x1=100+max_pad, y1=100, fillcolor="rgba(234,179,8,0.02)", line_width=0)
    fig.add_shape(type="rect", x0=100-max_pad, y0=100-max_pad, x1=100, y1=100, fillcolor="rgba(239,68,68,0.02)", line_width=0)
    fig.add_shape(type="rect", x0=100-max_pad, y0=100, x1=100, y1=100+max_pad, fillcolor="rgba(59,130,246,0.02)", line_width=0)
    
    fig.add_shape(type="line", x0=100, y0=100-max_pad, x1=100, y1=100+max_pad, line=dict(color="rgba(148,163,184,0.4)", width=1.5, dash="dash"))
    fig.add_shape(type="line", x0=100-max_pad, y0=100, x1=100+max_pad, y1=100, line=dict(color="rgba(148,163,184,0.4)", width=1.5, dash="dash"))
    
    for col in jdk_rs_ratio.columns:
        x_vals = jdk_rs_ratio[col].values
        y_vals = jdk_rs_momentum[col].values
        
        display_name = tickers.get(col, col).replace("Nifty ", "")
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines+markers', name=tickers.get(col, col), 
            line=dict(width=2.5),
            marker=dict(
                size=[4]*(len(x_vals)-1) + [12],
                symbol=['circle']*(len(x_vals)-1) + ['triangle-up'],
                line=dict(width=1, color="white")
            ),
            hovertemplate=f"<b>{tickers.get(col,col)}</b><br>RS Ratio: %{{x:.2f}}<br>RS Momentum: %{{y:.2f}}<extra></extra>"
        ))
        
        fig.add_annotation(
            x=x_vals[-1], y=y_vals[-1], text=f"<b>{display_name}</b>", 
            showarrow=False, xshift=14, yshift=6,
            font=dict(size=11, color="#0F172A", family="Arial Black"),
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="rgba(148,163,184,0.3)", borderpad=1, borderwidth=1, align="center"
        )
            
    fig.add_annotation(x=100+max_pad*0.75, y=100+max_pad*0.88, text="🟩 LEADING", font=dict(color="#16a34a", size=14, weight="bold"), showarrow=False)
    fig.add_annotation(x=100+max_pad*0.75, y=100-max_pad*0.88, text="🟨 WEAKENING", font=dict(color="#ca8a04", size=14, weight="bold"), showarrow=False)
    fig.add_annotation(x=100-max_pad*0.75, y=100-max_pad*0.88, text="🟥 LAGGING", font=dict(color="#dc2626", size=14, weight="bold"), showarrow=False)
    fig.add_annotation(x=100-max_pad*0.75, y=100+max_pad*0.88, text="🟦 IMPROVING", font=dict(color="#2563eb", size=14, weight="bold"), showarrow=False)
    
    fig.update_layout(
        title=dict(text=f"📅 Snapshot View Target Date/Time: <b>{title_date}</b>", font=dict(size=15, color="#475569")),
        xaxis_title="👉 Trend Strength (RS Ratio)", yaxis_title="🚀 Sector Velocity (RS Momentum)",
        xaxis=dict(range=x_range, gridcolor="rgba(241,245,249,1)", zeroline=False),
        yaxis=dict(range=y_range, gridcolor="rgba(241,245,249,1)", zeroline=False),
        height=850, margin=dict(l=20, r=30, t=50, b=20), plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# 🇮🇳 Complete National Stock Exchange (NSE) Sectoral Universe Map
nse_sectors_universe = {
    'GOVBI.NS': 'Nifty PSU Bank',
    '^NSEBANK': 'Nifty Bank',
    'CNXIT.NS': 'Nifty IT',
    'CNXAUTO.NS': 'Nifty Auto',
    'CNXPHARMA.NS': 'Nifty Pharma',
    'CNXFMCG.NS': 'Nifty FMCG',
    'CNXMETAL.NS': 'Nifty Metal',
    'CNXREALTY.NS': 'Nifty Realty',
    'CNXENERGY.NS': 'Nifty Energy',
    'CNXINFRA.NS': 'Nifty Infra',
    'CNXCOMMODITIES.NS': 'Nifty Commodities',
    'CNXCONSUMPTION.NS': 'Nifty Consumption',
    'CNXFINANCE.NS': 'Nifty Financial Services',
    'CNXMEDIA.NS': 'Nifty Media'
}

# Benchmark is Nifty 50 Index
nse_benchmark = '^NSEI'

# Sidebar Controls
st.sidebar.header("⚙️ NSE Configuration")

selected_sectors = st.sidebar.multiselect(
    "Select Indian Sectors to Plot",
    options=list(nse_sectors_universe.keys()), 
    default=list(nse_sectors_universe.keys()), 
    format_func=lambda x: nse_sectors_universe[x]
)

tail = st.sidebar.slider("Tail Length (History)", min_value=3, max_value=15, value=5)

# 📅 HISTORICAL SNAPSHOT OFFSET SLIDER
offset = st.sidebar.slider(
    "⏳ Historical Offset (Shift Backwards)", 
    min_value=0, max_value=7, value=0,
    help="Intraday (30m) me ye pichle 30-min candle chunks ko shift karega. Daily/Weekly me ye days/weeks shift karega."
)

if st.sidebar.button("🔄 Force Clear Cache", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

active_tickers = {k: nse_sectors_universe[k] for k in selected_sectors} if selected_sectors else nse_sectors_universe

# Viewport Tabs Setup WITH 30-MINUTE ENGINE AS TAB 1
t1, t2, t3, t4 = st.tabs(["⚡ 30-Min Intraday", "📈 Daily Matrix", "📆 Weekly Rotation", "⏳ Monthly Macro"])

with t1:
    r, m, s_date = calculate_rrg_cached(active_tickers, nse_benchmark, '30m', tail_length=tail, history_offset=offset)
    st.plotly_chart(plot_rrg_labeled(r, m, active_tickers, s_date), use_container_width=True)

with t2:
    r, m, s_date = calculate_rrg_cached(active_tickers, nse_benchmark, '1d', tail_length=tail, history_offset=offset)
    st.plotly_chart(plot_rrg_labeled(r, m, active_tickers, s_date), use_container_width=True)

with t3:
    r, m, s_date = calculate_rrg_cached(active_tickers, nse_benchmark, '1wk', tail_length=tail, history_offset=offset)
    st.plotly_chart(plot_rrg_labeled(r, m, active_tickers, s_date), use_container_width=True)

with t4:
    r, m, s_date = calculate_rrg_cached(active_tickers, nse_benchmark, '1mo', tail_length=tail, history_offset=offset)
    st.plotly_chart(plot_rrg_labeled(r, m, active_tickers, s_date), use_container_width=True)
