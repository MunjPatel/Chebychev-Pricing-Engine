import streamlit as st
import json
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from chebychev import ChebychevForecast
from scipy.stats import entropy, gaussian_kde
from datetime import datetime, timedelta
import os
from groq import Groq 

# -------------------------------
# 1. Page Configuration & CSS
# -------------------------------
st.set_page_config(
    layout="wide",
    page_title="Chebyshev Pricing Engine",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Professional Styling
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background-color: #1f2937; border: 1px solid #374151;
        padding: 15px; border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stMetricLabel"] { font-size: 13px; color: #9ca3af; font-weight: 500; }
    [data-testid="stMetricValue"] {
        font-size: 20px !important; /* Optimized for Bounds display */
        color: #f3f4f6;
        font-family: 'Source Code Pro', monospace;
        overflow-wrap: break-word;
        white-space: pre-wrap;
    }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #374151; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
    
    /* AI Box Styling */
    .stInfo {
        background-color: #1e293b;
        border: 1px solid #3b82f6;
        color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# 2. Helper Functions & State
# -------------------------------
@st.cache_data
def load_tickers():
    try:
        with open("sector_tickers.json", "r") as tick:
            return json.load(tick)
    except FileNotFoundError:
        return {}

def calculate_kl_divergence(p, q):
    p_norm = p / np.sum(p)
    q_norm = q / np.sum(q)
    return entropy(p_norm, q_norm)

# --- AI ANALYST FUNCTION ---
def generate_ai_memo(api_key, ticker, price, lower, upper, status, outliers, volatility):
    if not api_key:
        return "⚠️ API Key missing. Please enter it in the sidebar or secrets."
    
    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""
        You are a Senior Quantitative Analyst at Insight Partners. 
        Analyze this data for {ticker}:
        - Price: ${price:.2f}
        - Chebyshev Bounds (k=10): ${lower:.2f} - ${upper:.2f}
        - Regime: {status}
        - Volatility Spread: {volatility:.2f}%
        - Anomalies Detected: {len(outliers)}
        
        Write a "Flash Note" (max 100 words) for the Portfolio Manager.
        1. Interpret the regime (Stable vs Overbought/Oversold).
        2. Recommend an action (Hold, Mean Reversion Trade, or Hedge).
        3. Be concise and professional.
        """
        
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

# Initialize Session State
if 'analysis_data' not in st.session_state:
    st.session_state['analysis_data'] = None
if 'ticker_symbol' not in st.session_state:
    st.session_state['ticker_symbol'] = None

# -------------------------------
# 3. Sidebar & Navigation
# -------------------------------
tickers = load_tickers()
page = st.sidebar.radio("📍 Navigation", ["Dashboard", "Mathematical Framework"])

if page == "Dashboard":
    st.markdown("<h1 style='text-align: center; color: white;'>📊 Chebyshev Price Forecast</h1>", unsafe_allow_html=True)

    # --- Sidebar Inputs ---
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Asset Configuration")

    # --- API KEY LOGIC (Secrets -> Sidebar Fallback) ---
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("🔑 Groq API Key (for AI)", type="password")

    if tickers:
        sector = st.sidebar.selectbox("Sector", list(tickers.keys()))
        company_dict = tickers[sector]
        company_name = st.sidebar.selectbox("Asset", list(company_dict.keys()))
        ticker_symbol = company_dict[company_name]

        st.sidebar.caption(f"🔒 Model Parameter: **k = 10** (Fixed)")

        time_view = st.sidebar.radio(
            "Timeframe",
            ["Last Trading Session", "Full History"],
            index=0
        )
        
        st.sidebar.markdown("###")
        
        col_run, col_reset = st.sidebar.columns(2)
        
        with col_run:
            run_clicked = st.button("⚡ Run", type="primary", use_container_width=True)
            
        with col_reset:
            reset_clicked = st.button("🔄 Reset", type="secondary", use_container_width=True)

        # RESET LOGIC
        if reset_clicked:
            st.session_state['analysis_data'] = None
            st.session_state['ticker_symbol'] = None
            st.rerun()

        # RUN LOGIC
        if run_clicked:
            with st.spinner(f"Calculating Chebyshev bounds for {ticker_symbol}..."):
                try:
                    model = ChebychevForecast(ticker=ticker_symbol, k=10)
                    forecast_data, _ = model._forecast()
                    
                    st.session_state['analysis_data'] = forecast_data
                    st.session_state['ticker_symbol'] = ticker_symbol
                except Exception as e:
                    st.error(f"Computation Error: {e}")

    else:
        st.warning("Please upload 'sector_tickers.json' to proceed.")

    # --- Main Dashboard Render ---
    if st.session_state['analysis_data'] is not None:
        
        full_data = st.session_state['analysis_data']
        current_ticker = st.session_state['ticker_symbol']

        if time_view == "Last Trading Session":
            last_date = full_data.index[-1].date()
            plot_data = full_data[full_data.index.date == last_date].copy()
        else:
            plot_data = full_data.copy()

        # --- METRIC CALCULATIONS ---
        kl_div = calculate_kl_divergence(plot_data['Close'], plot_data['close_avg'])
        
        plot_data['spread_pct'] = ((plot_data['close_max'] - plot_data['close_min']) / plot_data['Close']) * 100
        avg_vol_spread = plot_data['spread_pct'].mean()

        last_row = plot_data.iloc[-1]
        current_price = last_row['Close']
        upper_b = last_row['close_max']
        lower_b = last_row['close_min']

        if current_price > upper_b:
            status = "⚠️ OVERBOUGHT"
            status_color = "red"
        elif current_price < lower_b:
            status = "⚠️ OVERSOLD"
            status_color = "red"
        else:
            status = "✅ STABLE"
            status_color = "green"

        # --- UI: Metrics Row ---
        st.markdown("---")
        # Fixed Column Ratio for Bounds Visibility
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1.5])
        with col1: st.metric("Asset Price", f"${current_price:.2f}")
        with col2: st.metric("KL Divergence", f"{kl_div:.2e}")
        with col3: st.metric("Avg Vol Spread", f"{avg_vol_spread:.2f}%")
        with col4: st.metric("99% Bounds", f"${lower_b:.2f} — ${upper_b:.2f}")
        st.markdown(f"**Market Regime:** :{status_color}[**{status}**]")

        # --- OUTLIER DETECTION ---
        outliers = plot_data[~plot_data['in_range']].copy()
        has_breaches = not outliers.empty

        # --- AI ANALYST SECTION ---
        st.markdown("###")
        if st.button("🤖 Generate AI Analyst Memo"):
            with st.spinner("Consulting AI Model..."):
                memo = generate_ai_memo(
                    api_key=api_key,
                    ticker=current_ticker,
                    price=current_price,
                    lower=lower_b,
                    upper=upper_b,
                    status=status,
                    outliers=outliers,
                    volatility=avg_vol_spread
                )
                st.info(f"**AI Flash Note:**\n\n{memo}")

        # --- DYNAMIC TABS ---
        tab_names = ["📈 Time Series", "🔔 Distribution (KDE)", "📉 Correlation Matrix"]
        if has_breaches:
            tab_names.append("⚠️ Breach Analysis")
        
        tabs = st.tabs(tab_names)

        # ==================================================
        # TAB 1: TIME SERIES
        # ==================================================
        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['close_min'], mode="lines", line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['close_max'], mode="lines", line=dict(width=0), fill='tonexty', fillcolor='rgba(46, 204, 113, 0.15)', name="99% Confidence"))
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['close_avg'], mode="lines", line=dict(color='orange', dash='dash', width=1), name="Forecast Mean"))
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'], mode="lines", line=dict(color='#F8FAFC', width=2), name="Actual Price"))
            
            if has_breaches:
                fig.add_trace(go.Scatter(x=outliers.index, y=outliers['Close'], mode="markers", marker=dict(color='#EF4444', size=6, symbol='x'), name="Breach"))

            fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#374151'), hovermode="x unified", legend=dict(orientation="h", y=1.05, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

        # ==================================================
        # TAB 2: DISTRIBUTION
        # ==================================================
        with tabs[1]:
            st.markdown(f"##### Kernel Density Estimation for {ticker_symbol}")
            clean_close = plot_data['Close'].dropna()
            clean_avg = plot_data['close_avg'].dropna()
            kde_close = gaussian_kde(clean_close)
            kde_avg = gaussian_kde(clean_avg)
            x_range = np.linspace(min(clean_close.min(), clean_avg.min()), max(clean_close.max(), clean_avg.max()), 200)

            fig_kde = go.Figure()
            fig_kde.add_trace(go.Scatter(x=x_range, y=kde_avg(x_range), mode='lines', fill='tozeroy', name='Forecast Model', line=dict(color='orange', dash='dot', width=1), fillcolor='rgba(255, 165, 0, 0.1)'))
            fig_kde.add_trace(go.Scatter(x=x_range, y=kde_close(x_range), mode='lines', fill='tozeroy', name='Actual Price', line=dict(color='#3b82f6', width=2), fillcolor='rgba(59, 130, 246, 0.1)'))

            fig_kde.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Price ($)", yaxis_title="Density", xaxis=dict(showgrid=False, gridcolor='#374151'), yaxis=dict(showgrid=True, gridcolor='#374151'), legend=dict(orientation="h", y=1.05, xanchor="right", x=1))
            st.plotly_chart(fig_kde, use_container_width=True)

        # ==================================================
        # TAB 3: CORRELATION MAP
        # ==================================================
        with tabs[2]:
            st.markdown(f"##### Feature Correlation Matrix for {ticker_symbol}")
            numeric_cols = ['Close', 'close_min', 'close_max', 'close_avg', 'price_std', 'price_mu']
            available_cols = [c for c in numeric_cols if c in plot_data.columns]
            
            if len(available_cols) > 1:
                corr_matrix = plot_data[available_cols].corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                    colorscale='RdBu_r', zmin=-1, zmax=1,
                    text=corr_matrix.values.round(2), texttemplate="%{text}", showscale=True
                ))
                fig_corr.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title="Pearson Correlation Coefficients")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Not enough numeric data available for correlation analysis.")

        # ==================================================
        # TAB 4: BREACH ANALYSIS (CONDITIONAL)
        # ==================================================
        if has_breaches:
            with tabs[3]:
                st.markdown(f"##### ⚠️ Breach Severity Analysis for {ticker_symbol}")
                outliers['deviation_amt'] = np.where(
                    outliers['Close'] > outliers['close_max'],
                    outliers['Close'] - outliers['close_max'],
                    outliers['close_min'] - outliers['Close']
                )
                
                col_b_plot, col_b_data = st.columns([2, 1])
                with col_b_plot:
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(x=outliers.index, y=outliers['deviation_amt'], marker_color='#EF4444', name='Deviation ($)'))
                    fig_bar.update_layout(title="Magnitude of Breaches ($)", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(showgrid=True, gridcolor='#374151'), showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                with col_b_data:
                    st.markdown("###### Breach Log")
                    export_df = outliers[['Close', 'close_min', 'close_max', 'deviation_amt']].copy()
                    export_df.columns = ['Price', 'Lower', 'Upper', 'Dev ($)']
                    st.dataframe(export_df.style.format("{:.2f}"), height=200, use_container_width=True)
                    
                    csv_breach = export_df.to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 Download Breach Report",
                        data=csv_breach,
                        file_name=f"{current_ticker}_breaches.csv",
                        mime="text/csv",
                        type="primary",
                        key="breach_dl_btn"
                    )

        # --- General Data Export ---
        st.markdown("###")
        csv_full = plot_data.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Full Analysis CSV",
            data=csv_full,
            file_name=f"{current_ticker}_chebyshev_full.csv",
            mime="text/csv",
            key="full_dl_btn"
        )

# ==============================================================
# PAGE 2: MATH FRAMEWORK
# ==============================================================
elif page == "Mathematical Framework":
    st.title("📘 Derivation of Chebyshev Price Bounds")

    st.markdown("### 1. The Log-Normal Assumption")
    st.markdown(
        r"""
        Financial prices ($P$) are strictly positive and often skewed. We transform them into **log-space** to approximate a symmetric distribution suitable for moment analysis.

        Let $L = \ln(P)$, where:
        * $\mu_L = E[L]$ (Mean of log-price)
        * $\sigma_L = \sqrt{\text{Var}(L)}$ (Std Dev of log-price)
        """
    )

    st.markdown("### 2. Chebyshev's Inequality")
    st.markdown(
        r"""
        Chebyshev's inequality states that for *any* probability distribution (with finite variance), the probability of a value lying more than $k$ standard deviations from the mean is at most $1/k^2$.

        $$
        P(|L - \mu_L| \ge k\sigma_L) \le \frac{1}{k^2}
        $$

        Conversely, the probability of remaining **within** $k$ standard deviations is:

        $$
        P(|L - \mu_L| \le k\sigma_L) \ge 1 - \frac{1}{k^2}
        $$
        """
    )

    st.markdown("---")

    st.markdown("### 3. Algebraic Derivation of Price Bounds")
    st.markdown("We solve the inequality for the price $P$ in four steps:")

    col_math_L, col_math_R = st.columns([2, 1.5])

    with col_math_L:
        st.markdown("**Step A: Expand the Modulus**")
        st.latex(r"-k\sigma_L \le L - \mu_L \le k\sigma_L")

        st.markdown("**Step B: Isolate the Log-Variable ($L$)**")
        st.latex(r"\mu_L - k\sigma_L \le L \le \mu_L + k\sigma_L")

        st.markdown("**Step C: Substitute Log Definition ($L = \ln P$)**")
        st.latex(r"\mu_L - k\sigma_L \le \ln P \le \mu_L + k\sigma_L")

        st.markdown("**Step D: Exponentiate to find Price ($P$)**")
        st.latex(r"\boxed{e^{\mu_L - k\sigma_L} \le P \le e^{\mu_L + k\sigma_L}}")

    with col_math_R:
        st.info(
            """
            **Why is this useful?**

            Unlike Bollinger Bands, the price bounds derived here do not require the data to be Normally distributed. 

            They provide a "distribution-free" upper and lower bound on the price of the asset.
            """
        )
        st.markdown(
            """
            | k | Confidence Interval ($1 - 1/k^2$) |
            | :---: | :--- |
            | 2 | 75.00% |
            | 5 | 96.00% |
            | **10** | **99.00% (Black Swan Safe)** |
            """
        )
        st.caption("In this model, we fix k=10 to capture extreme market events.")

    st.markdown("---")

    st.markdown("### 4. KL Divergence (Relative Entropy)")
    st.markdown(
        r"""
        To measure the efficiency of our Forecast Mean against the Actual Price action, we utilize **Kullback-Leibler (KL) Divergence**. 

        It measures how one probability distribution $P$ (Actual Prices) diverges from a second, expected probability distribution $Q$ (Forecast Model).

        $$
        D_{KL}(P || Q) = \sum_{x \in X} P(x) \log \left( \frac{P(x)}{Q(x)} \right)
        $$

        * **Low Divergence ($\approx 0$):** The forecast distribution closely matches the actual market behavior (High Predictability).
        * **High Divergence:** The market is behaving erratically compared to the forecast model (Regime Shift or High Volatility).
        """
    )