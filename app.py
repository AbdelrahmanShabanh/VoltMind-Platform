

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, time

# ────────────────────────────── Page Configuration ──────────────────────────
st.set_page_config(
    page_title="Grid Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None,
)

# ────────────────────────────── Theme Variables ──────────────────────────────
DARK_BG = "#0f1419"
CARD_BG = "rgba(20, 30, 48, 0.6)"
ACCENT_CYAN = "#00d9ff"
ACCENT_PURPLE = "#a855f7"
ACCENT_GREEN = "#10b981"
TEXT_PRIMARY = "#f0f4f8"
TEXT_SECONDARY = "#94a3b8"
BORDER_COLOR = "rgba(0, 217, 255, 0.1)"

# ────────────────────────────── Advanced CSS Styling ────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;600;700;800&display=swap');
    
    :root {{
        --bg-dark: {DARK_BG};
        --card-bg: {CARD_BG};
        --accent-cyan: {ACCENT_CYAN};
        --accent-purple: {ACCENT_PURPLE};
        --accent-green: {ACCENT_GREEN};
        --text-primary: {TEXT_PRIMARY};
        --text-secondary: {TEXT_SECONDARY};
        --border: {BORDER_COLOR};
    }}
    
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        background: linear-gradient(135deg, {DARK_BG} 0%, #1a2a3a 100%) !important;
        color: {TEXT_PRIMARY} !important;
        font-family: 'Poppins', sans-serif !important;
    }}
    
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(15, 20, 25, 0.95) 0%, rgba(20, 30, 50, 0.9) 100%) !important;
        border-right: 1px solid var(--border) !important;
        backdrop-filter: blur(20px) !important;
    }}
    
    [data-testid="stSidebar"] * {{ color: var(--text-primary) !important; }}
    
    .glass-card {{
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 28px;
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.05);
    }}
    
    .glass-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(0, 217, 255, 0.12);
        border-color: rgba(0, 217, 255, 0.2);
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 24px 16px;
        text-align: center;
        transition: all 0.5s cubic-bezier(0.23, 1, 0.320, 1);
        animation: float 4s ease-in-out infinite;
    }}
    
    .metric-card:nth-child(1) {{ animation-delay: 0s; }}
    .metric-card:nth-child(2) {{ animation-delay: 0.15s; }}
    .metric-card:nth-child(3) {{ animation-delay: 0.3s; }}
    .metric-card:nth-child(4) {{ animation-delay: 0.45s; }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
    }}
    
    /* Emoji icons are now static by default (no continuous spin). 
       A gentle scale/lift only happens on hover for a subtle, controlled effect. */
    .metric-icon {{
        font-size: 2.4rem;
        margin-bottom: 8px;
        display: inline-block;
        transition: transform 0.35s cubic-bezier(0.23, 1, 0.320, 1);
    }}
    
    .metric-card:hover .metric-icon {{ transform: scale(1.15) translateY(-4px); }}
    
    .metric-label {{
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }}
    
    .metric-value {{
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .hero {{ text-align: center; padding: 48px 0 32px; animation: fadeInDown 0.8s ease-out; }}
    
    @keyframes fadeInDown {{ from {{ opacity: 0; transform: translateY(-30px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    
    .hero-title {{
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -2px;
        line-height: 1.1;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple), var(--accent-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 12px;
    }}
    
    .hero-subtitle {{
        font-size: 1.1rem;
        color: var(--text-secondary);
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.7;
        font-weight: 300;
    }}
    
    .prediction-box {{
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border: 2px solid var(--accent-cyan);
        border-radius: 28px;
        padding: 48px 32px;
        text-align: center;
        animation: scaleIn 0.6s cubic-bezier(0.23, 1, 0.320, 1), pulse 2.5s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }}
    
    @keyframes scaleIn {{ 0% {{ opacity: 0; transform: scale(0.9); }} 100% {{ opacity: 1; transform: scale(1); }} }}
    @keyframes pulse {{ 0%, 100% {{ box-shadow: 0 0 20px rgba(0, 217, 255, 0.15); }} 50% {{ box-shadow: 0 0 50px rgba(0, 217, 255, 0.3); }} }}
    
    .prediction-box::before {{
        content: ''; position: absolute; top: -50%; right: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(0, 217, 255, 0.1) 0%, transparent 70%);
        animation: shimmer 3s ease-in-out infinite;
    }}
    
    @keyframes shimmer {{ 0%, 100% {{ transform: translate(0, 0); }} 50% {{ transform: translate(20px, 20px); }} }}
    
    .result-label {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 3px; color: var(--text-secondary); margin-bottom: 12px; font-weight: 600; }}
    .result-value {{ font-size: 4.2rem; font-weight: 900; background: linear-gradient(135deg, var(--accent-cyan), var(--accent-green)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1; margin: 12px 0; }}
    .result-unit {{ font-size: 1.5rem; font-weight: 600; color: var(--accent-cyan); letter-spacing: 1px; }}
    
    .demand-tag {{
        display: inline-block; margin-top: 20px; padding: 10px 24px; border-radius: 50px;
        font-size: 0.85rem; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase;
        animation: fadeInUp 0.6s ease-out 0.3s both;
    }}
    
    @keyframes fadeInUp {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    
    .tag-low {{ background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.05) 100%); color: var(--accent-green); border: 1.5px solid var(--accent-green); box-shadow: 0 0 20px rgba(16, 185, 129, 0.1); }}
    .tag-medium {{ background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.05) 100%); color: #f59e0b; border: 1.5px solid #f59e0b; box-shadow: 0 0 20px rgba(245, 158, 11, 0.1); }}
    .tag-high {{ background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.05) 100%); color: #ef4444; border: 1.5px solid #ef4444; box-shadow: 0 0 20px rgba(239, 68, 68, 0.1); }}
    
    .divider {{ height: 1px; background: linear-gradient(90deg, transparent, var(--border), transparent); margin: 40px 0; border: none; }}
    
    div[data-baseweb="input"] input, div[data-baseweb="select"] div, .stSlider [role="slider"] {{
        background: rgba(30, 40, 60, 0.8) !important; color: var(--text-primary) !important;
        border: 1px solid var(--border) !important; border-radius: 14px !important; transition: all 0.3s ease !important;
    }}
    div[data-baseweb="input"] input:focus, div[data-baseweb="select"] div:focus {{ border-color: var(--accent-cyan) !important; box-shadow: 0 0 20px rgba(0, 217, 255, 0.2) !important; }}
    .stSlider [role="slider"] {{ background: var(--accent-cyan) !important; }}
    label, .stSelectbox label, .stSlider label, .stNumberInput label {{ color: var(--text-primary) !important; font-weight: 600 !important; font-size: 0.95rem !important; }}
    
    .sidebar-title {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2.5px; color: var(--accent-cyan); margin: 28px 0 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); font-weight: 700; }}
    .sidebar-icon {{ font-size: 1.3rem; margin-right: 6px; }}
    
    .insight-box {{
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
        border-left: 4px solid var(--accent-cyan); border-radius: 16px; padding: 24px 28px; margin-top: 24px;
        animation: slideInLeft 0.6s ease-out 0.5s both;
    }}
    
    @keyframes slideInLeft {{ from {{ opacity: 0; transform: translateX(-30px); }} to {{ opacity: 1; transform: translateX(0); }} }}
    .insight-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; color: var(--text-secondary); margin-bottom: 8px; font-weight: 700; }}
    .insight-text {{ font-size: 1rem; color: var(--text-primary); line-height: 1.7; font-weight: 400; }}
    
    [data-testid="stHeader"] {{ background: transparent !important; }}
    .footer {{ text-align: center; font-size: 0.75rem; color: var(--text-secondary); padding: 24px 0; border-top: 1px solid var(--border); margin-top: 40px; }}
    
    @media (max-width: 768px) {{ .hero-title {{ font-size: 2.2rem; }} .result-value {{ font-size: 3rem; }} .metric-card {{ padding: 16px 12px; }} }}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────── Load Trained Model ──────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str = "xgboost_smartgrid_model.pkl"):
    """Load XGBoost model from pickle file (cached)."""
    return joblib.load(path)

model = load_model()

# ────────────────────────────── Constants & Configuration ───────────────────
# EXTREMELY IMPORTANT: This exactly matches the error message feature order!
FEATURE_ORDER = [
    "rolling_mean_3",
    "lag_1_power",
    "lag_2_power",
    "avg_voltage",
    "minute",
    "hour",
    "dayofweek",
    "is_weekend",
]

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ────────────────────────────── Helper Functions ──────────────────────────
def classify_load(kw: float) -> tuple[str, str, str]:
    if kw < 1.5: return "Low Demand", "tag-low", "🟢"
    elif kw < 3.5: return "Moderate Demand", "tag-medium", "🟡"
    return "High Demand", "tag-high", "🔴"
# bya5od numeric data y7wlha graphs
def build_gauge_chart(value: float, max_val: float = 6.0) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value, 3),
        number={"font": {"size": 56, "color": ACCENT_CYAN, "family": "Outfit"}, "suffix": " kW"},
        gauge={
            "axis": {"range": [0, max_val], "tickwidth": 2, "tickcolor": "#475569", "dtick": max_val / 6, "tickfont": {"size": 12, "color": TEXT_SECONDARY, "family": "Poppins"}},
            "bar": {"color": ACCENT_CYAN, "thickness": 0.4},
            "bgcolor": "rgba(30, 40, 60, 0.4)", "borderwidth": 2, "bordercolor": ACCENT_CYAN,
            "steps": [
                {"range": [0, max_val * 0.25], "color": "rgba(16, 185, 129, 0.08)"},
                {"range": [max_val * 0.25, max_val * 0.58], "color": "rgba(245, 158, 11, 0.08)"},
                {"range": [max_val * 0.58, max_val], "color": "rgba(239, 68, 68, 0.08)"},
            ],
            "threshold": {"line": {"color": ACCENT_PURPLE, "width": 4}, "thickness": 0.9, "value": value},
        },
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=20, l=40, r=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"family": "Poppins", "color": TEXT_PRIMARY})
    return fig

def build_radar_chart(data: dict) -> go.Figure:
    labels, raw = list(data.keys()), list(data.values())
    arr = np.array(raw, dtype=float)
    lo, hi = arr.min(), arr.max()
    norm = ((arr - lo) / (hi - lo + 1e-9)).tolist()
    norm += [norm[0]]
    labels_closed = labels + [labels[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm, theta=labels_closed, fill="toself", fillcolor="rgba(0, 217, 255, 0.15)",
        line=dict(color=ACCENT_CYAN, width=2.5), hovertemplate="%{theta}: %{customdata:.3f}<extra></extra>",
        customdata=raw + [raw[0]], name="Features",
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(0, 217, 255, 0.06)", tickfont=dict(size=10, color=TEXT_SECONDARY)),
            angularaxis=dict(gridcolor="rgba(0, 217, 255, 0.08)", tickfont=dict(size=11, color=TEXT_SECONDARY, family="Poppins")),
        ),
        showlegend=False, height=350, margin=dict(t=50, b=50, l=70, r=70), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins", color=TEXT_PRIMARY),
    )
    return fig

def build_bar_chart(data: dict) -> go.Figure:
    labels, values = list(data.keys()), list(data.values())
    colors = [ACCENT_CYAN, ACCENT_PURPLE, ACCENT_GREEN, "#f59e0b", "#ef4444", "#06b6d4", "#8b5cf6", "#ec4899", "#3b82f6"]
    
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker=dict(color=colors[:len(labels)], line=dict(width=0), cornerradius=8),
        hovertemplate="%{y}: <b>%{x:.3f}</b><extra></extra>", text=[f"{v:.2f}" for v in values],
        textposition="outside", textfont=dict(color=TEXT_PRIMARY, size=11, family="Poppins"),
    ))
    
    fig.update_layout(
        height=350, margin=dict(t=20, b=20, l=100, r=60), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(0, 217, 255, 0.04)", tickfont=dict(color=TEXT_SECONDARY, size=11, family="Poppins"), title=None, showgrid=True),
        yaxis=dict(tickfont=dict(color=TEXT_SECONDARY, size=11, family="Poppins"), title=None),
        font=dict(family="Poppins", color=TEXT_PRIMARY), showlegend=False,
    )
    return fig

# ────────────────────────────── Sidebar Configuration ──────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:20px 0 10px;">
        <div style="font-size:3rem; margin-bottom:8px;">⚡</div>
        <h2 style="margin:0; font-weight:800; font-size:1.8rem; background:linear-gradient(135deg, {0}, {1}); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Grid Predictor
        </h2>
        <p style="font-size:0.8rem; color:#94a3b8; margin:4px 0 0;">Real-time Power Forecasting</p>
    </div>
    <div class="divider"></div>
    """.format(ACCENT_CYAN, ACCENT_PURPLE), unsafe_allow_html=True)
    
    # ── 1. Household Selection (The missing feature) ──
    st.markdown('<p class="sidebar-title"><span class="sidebar-icon">🏠</span>Target Area</p>', unsafe_allow_html=True)
    meter_encoded = st.selectbox(
        "Select Household / Meter",
        options=[0, 1, 2, 3, 4, 5],
        format_func=lambda x: f"Household {x}",
        help="Select the specific household ID for the prediction."
    )
    
    # ── 2. Electrical readings ──
    st.markdown('<p class="sidebar-title"><span class="sidebar-icon">⚙️</span>Electrical Readings</p>', unsafe_allow_html=True)
    avg_voltage = st.slider("Average Voltage (V)", min_value=200.0, max_value=260.0, value=230.0, step=0.5)
    
    # ── 3. Temporal features ──
    st.markdown('<p class="sidebar-title"><span class="sidebar-icon">🕒</span>Time & Date</p>', unsafe_allow_html=True)
    input_time = st.time_input("Time of Day", value=time(14, 30))
    day_of_week = st.selectbox("Day of Week", options=list(range(7)), format_func=lambda d: f"{DAY_NAMES[d]}  {'📅' if d < 5 else '🌴'}", index=0)
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # ── 4. Lag features ──
    st.markdown('<p class="sidebar-title"><span class="sidebar-icon">📊</span>Historical Data</p>', unsafe_allow_html=True)
    lag_1 = st.slider("Lag-1 Power (kW)", min_value=0.0, max_value=12.0, value=1.2, step=0.01)
    lag_2 = st.slider("Lag-2 Power (kW)", min_value=0.0, max_value=12.0, value=1.1, step=0.01)
    
    # Auto-calculated rolling mean
    rolling_mean = (lag_1 + lag_2) / 2.0
    st.markdown(f"""
    <div style="background: rgba(30, 40, 60, 0.8); border: 1px solid var(--border); border-radius: 14px; padding: 12px 14px; margin-top: 10px;">
        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 4px; font-weight: 600;">Rolling Mean-3 (Auto)</div>
        <div style="font-size: 1.2rem; color: var(--accent-cyan); font-weight: 700;">{rolling_mean:.2f} kW</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider" style="margin:20px 0;"></div>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center; font-size:0.7rem; color:{TEXT_SECONDARY}; line-height:1.6;"><strong>Model: XGBoost</strong><br>Features: 9 | Target: Active Power</p>', unsafe_allow_html=True)

# ────────────────────────────── Main Content Area ───────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-title">⚡ Smart Grid Load Predictor</div>
    <div class="hero-subtitle">
        The system is completely reactive. Try adjusting the parameters on the left 
        and watch the AI re-calculate predictions instantly in real-time.
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Summary Cards ──
col1, col2, col3, col4 = st.columns(4, gap="medium")
card_items = [
    ("🔌", "Voltage", f"{avg_voltage:.1f} V"),
    ("⏰", "Time", f"{input_time.strftime('%H:%M')}"),
    ("📅", "Day", DAY_NAMES[day_of_week]),
    ("📈", "Rolling Mean", f"{rolling_mean:.2f} kW"),
]

for col, (icon, label, value) in zip([col1, col2, col3, col4], card_items):
    col.markdown(f"""
    <div class="glass-card metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ────────────────────────────── REAL-TIME PREDICTION ───────────────────────────
# We removed the predict button The app now updates automatically and instantly

raw_features = {
    "meter_encoded": meter_encoded,
    "avg_voltage": avg_voltage,
    "hour": input_time.hour,
    "minute": input_time.minute,
    "dayofweek": day_of_week,
    "is_weekend": is_weekend,
    "lag_1_power": lag_1,
    "lag_2_power": lag_2,
    "rolling_mean_3": rolling_mean,
}

# Create dataframe with exact feature order
input_df = pd.DataFrame({k: [v] for k, v in raw_features.items()})
input_df = input_df[FEATURE_ORDER]

try:
    # Make prediction automatically
    prediction = float(model.predict(input_df)[0])
    demand_label, demand_class, demand_emoji = classify_load(prediction)
    
    # Display prediction result
    st.markdown(f"""
    <div class="glass-card prediction-box">
        <div class="result-label">📊 Predicted Active Power</div>
        <div class="result-value">{prediction:.2f}</div>
        <div class="result-unit">kilowatts</div>
        <div class="demand-tag {demand_class}">
            {demand_emoji} {demand_label}
        </div>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    # Analytics charts header
    st.markdown(f"""
    <div class="glass-card" style="padding:16px 20px 8px; background: rgba(0,0,0,0.2);">
        <p style="text-align:center; font-size:0.75rem; text-transform:uppercase; letter-spacing:2px; color:{TEXT_SECONDARY}; margin:0; font-weight:700;">
           📊 Live Visualization Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Charts in columns
    gauge_col, radar_col = st.columns(2, gap="medium")
    
    with gauge_col:
        st.plotly_chart(build_gauge_chart(prediction), use_container_width=True, config={"displayModeBar": False})
    
    with radar_col:
        st.plotly_chart(build_radar_chart(raw_features), use_container_width=True, config={"displayModeBar": False})
    
    # Feature values bar chart
    st.plotly_chart(build_bar_chart(raw_features), use_container_width=True, config={"displayModeBar": False})
    
    # Smart recommendation
    if prediction < 1.5:
        icon, title, message = "💚", "Low Demand", "Grid load is minimal. Excellent opportunity to reduce generation, shift maintenance tasks, or optimize battery storage."
    elif prediction < 3.5:
        icon, title, message = "⚙️", "Moderate Demand", "Power consumption is stable. Monitor trends closely for any sudden spikes. Current capacity handles load efficiently."
    else:
        icon, title, message = "⚠️", "High Demand Alert", "Grid is under heavy load! Activate peaking plants immediately and implement demand-response programs to prevent outages."
    
    st.markdown(f"""
    <div class="glass-card insight-box">
        <div class="insight-label">{icon} {title}</div>
        <div class="insight-text">{message}</div>
    </div>
    """, unsafe_allow_html=True)
    
except Exception as exc:
    st.error(f"❌ **Prediction Error:** {str(exc)}")

# Footer
st.markdown(f"""
<div class="footer">
    <p>Powered by XGBoost Machine Learning | Reactive Streamlit Architecture</p>
    <p>© 2026 Smart Grid Predictor | Real-time Power Consumption Analytics</p>
</div>
""", unsafe_allow_html=True)
