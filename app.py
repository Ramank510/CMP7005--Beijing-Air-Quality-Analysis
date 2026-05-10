import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle

st.set_page_config(
    page_title="Beijing Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #0e1117;}
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label {
        color: #a0aec0 !important;
        font-size: 13px !important;
    }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #ffffff !important;
        font-size: 24px !important;
        font-weight: bold !important;
    }
    h1 {color: #ffffff !important; font-size: 2rem !important;}
    h2 {color: #e2e8f0 !important;}
    h3 {color: #cbd5e0 !important;}
    section[data-testid="stSidebar"] {
        background-color: #1a1f2e;
        border-right: 1px solid #2d3250;
    }
    div[data-testid="stInfo"] {background-color: #1e3a5f; border-left: 4px solid #3b82f6; border-radius: 5px;}
    div[data-testid="stSuccess"] {background-color: #1a3a2a; border-left: 4px solid #22c55e; border-radius: 5px;}
    div[data-testid="stError"] {background-color: #3a1a1a; border-left: 4px solid #ef4444; border-radius: 5px;}
    div[data-testid="stWarning"] {background-color: #3a2a1a; border-left: 4px solid #f59e0b; border-radius: 5px;}
    button[data-baseweb="tab"] {font-size: 14px !important; font-weight: 600 !important;}
    hr {border-color: #2d3250;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_dataset.csv")

@st.cache_resource
def load_model():
    with open("data/rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("data/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("data/features.pkl", "rb") as f:
        features = pickle.load(f)
    with open("data/metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    return model, scaler, features, metrics

df = load_data()
model, scaler, features, metrics = load_model()

def draw_gauge(value, max_val=300):
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={"projection": "polar"}, facecolor='#1e2130')
    zones = [
        (0, 12, "#22c55e", "Good"),
        (12, 35, "#eab308", "Moderate"),
        (35, 55, "#f97316", "Unhealthy(S)"),
        (55, 150, "#ef4444", "Unhealthy"),
        (150, 250, "#a855f7", "Very Unhealthy"),
        (250, 300, "#1e293b", "Hazardous")
    ]
    for start, end, color, label in zones:
        theta_start = np.pi * (1 - start/max_val)
        theta_end = np.pi * (1 - end/max_val)
        ax.barh(1, theta_start - theta_end, left=theta_end, height=0.5, color=color, alpha=0.9)
    angle = np.pi * (1 - min(value, max_val)/max_val)
    ax.annotate("", xy=(angle, 1), xytext=(angle, 0), arrowprops=dict(arrowstyle="->", color="white", lw=3))
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.5)
    ax.set_xlim(0, np.pi)
    ax.axis("off")
    ax.set_facecolor('#1e2130')
    ax.set_title("PM2.5: " + str(round(value, 1)) + " µg/m³\n(WHO Safe Limit: 15 µg/m³)",
                 fontsize=13, fontweight="bold", pad=20, color='white')
    patches = [mpatches.Patch(color=c, label=l) for _, _, c, l in zones]
    ax.legend(handles=patches, loc="lower center", ncol=3, fontsize=8,
              bbox_to_anchor=(0.5, -0.15), facecolor='#1e2130', labelcolor='white', edgecolor='#2d3250')
    fig.patch.set_facecolor('#1e2130')
    return fig

with st.sidebar:
    st.markdown("## 🌫️ Beijing Air Quality")
    st.markdown("**CMP7005 — PRAC1**")
    st.markdown("---")
    page = st.radio("Navigate to:", ["🏠 Overview", "📊 Dataset Explorer", "📈 Visualisations", "🤖 Model Outputs"])
    st.markdown("---")
    st.markdown("### 📍 Monitoring Stations")
    st.markdown("🔴 **Dongsi** — Urban")
    st.markdown("🟠 **Guanyuan** — Urban")
    st.markdown("🟢 **Dingling** — Suburban")
    st.markdown("🔵 **Huairou** — Suburban")
    st.markdown("---")
    st.markdown("### 📅 Dataset Info")
    st.markdown("**Period:** Mar 2013 — Feb 2017")
    st.markdown("**Records:** 140,256 hourly readings")
    st.markdown("**Pollutants:** PM2.5, PM10, SO2, NO2, CO, O3")

if page == "🏠 Overview":
    st.title("🌫️ Beijing Air Quality Dashboard")
    st.markdown("Comprehensive analysis of hourly air quality data from **4 monitoring stations** across Beijing between **March 2013 and February 2017**.")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Total Records", f"{len(df):,} hours")
    col2.metric("🌡️ Avg PM2.5", f"{df['PM2.5'].mean():.1f} µg/m³",
                delta=f"{df['PM2.5'].mean()-15:.1f} above WHO limit", delta_color="inverse")
    col3.metric("⚠️ Max PM2.5", f"{df['PM2.5'].max():.0f} µg/m³")
    col4.metric("✅ WHO Safe Limit", "15 µg/m³")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🎯 Air Quality Gauge")
        avg_pm25 = df["PM2.5"].mean()
        fig = draw_gauge(avg_pm25)
        st.pyplot(fig, use_container_width=True)
        st.caption(f"⚠️ Beijing's average PM2.5 is **{avg_pm25/15:.1f}x** above the WHO safe limit")
    with col2:
        st.subheader("📊 Key Statistics")
        st.markdown("---")
        good = len(df[df["AQI_Category"] == "1-Good"])
        haz = len(df[df["AQI_Category"] == "6-Hazardous"])
        total = len(df)
        col_a, col_b = st.columns(2)
        col_a.metric("🟢 Good Air Quality Hours", f"{good:,}", f"{good/total*100:.1f}% of total")
        col_b.metric("⚫ Hazardous Air Hours", f"{haz:,}", f"{haz/total*100:.1f}% of total", delta_color="inverse")
        months_full = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                       7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
        worst_m = df.groupby("month")["PM2.5"].mean().idxmax()
        best_m = df.groupby("month")["PM2.5"].mean().idxmin()
        col_a2, col_b2 = st.columns(2)
        col_a2.metric("📅 Worst Month", months_full[worst_m])
        col_b2.metric("📅 Best Month", months_full[best_m])

    st.markdown("---")
    st.subheader("📋 Full Station Summary")
    summary = df.groupby("station").agg(
        Records=("PM2.5", "count"),
        Avg_PM25=("PM2.5", "mean"),
        Max_PM25=("PM2.5", "max"),
        Min_PM25=("PM2.5", "min"),
        Station_Type=("station_type", "first")
    ).round(2).reset_index()
    summary.columns = ["Station", "Records", "Avg PM2.5", "Max PM2.5", "Min PM2.5", "Type"]
    st.dataframe(summary, use_container_width=True, hide_index=True)

elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.markdown("Browse, filter and download the Beijing air quality dataset.")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        sf = st.selectbox("🏙️ Station", ["All"] + sorted(df["station"].unique().tolist()))
    with col2:
        yf = st.selectbox("📅 Year", ["All"] + sorted(df["year"].unique().tolist()))
    with col3:
        ssf = st.selectbox("🌿 Season", ["All", "Winter", "Spring", "Summer", "Autumn"])

    rng = st.slider("PM2.5 Range", min_value=2, max_value=881, value=(2, 881))

    filtered = df.copy()
    if sf != "All": filtered = filtered[filtered["station"] == sf]
    if yf != "All": filtered = filtered[filtered["year"] == int(yf)]
    if ssf != "All": filtered = filtered[filtered["season"] == ssf]
    filtered = filtered[(filtered["PM2.5"] >= rng[0]) & (filtered["PM2.5"] <= rng[1])]

    col1, col2, col3 = st.columns(3)
    col1.metric("📋 Records Shown", f"{len(filtered):,}")
    col2.metric("📊 Avg PM2.5", f"{filtered['PM2.5'].mean():.1f} µg/m³")
    col3.metric("⚠️ Max PM2.5", f"{filtered['PM2.5'].max():.0f} µg/m³")

    st.dataframe(filtered[["datetime","station","station_type","PM2.5","PM10","NO2","SO2","CO","O3","TEMP","AQI_Category","season"]].head(200),
                 use_container_width=True, hide_index=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Dataset as CSV", data=csv,
                       file_name="beijing_air_quality_filtered.csv", mime="text/csv", use_container_width=True)

elif page == "📈 Visualisations":
    st.title("📈 Visualisations")
    palette = {'Dongsi':'#ef4444','Guanyuan':'#f97316','Dingling':'#22c55e','Huairou':'#3b82f6'}
    chart = st.selectbox("📊 Select a chart:", [
        "PM2.5 Distribution by Station","Monthly PM2.5 Trend","Hourly Pollution Pattern",
        "PM2.5 Heatmap (Hour vs Month)","Weekend vs Weekday Comparison","Yearly PM2.5 Trend",
        "Correlation Heatmap","Urban vs Suburban Comparison"
    ])
    stations = st.multiselect("🏙️ Select stations:", df["station"].unique().tolist(), default=df["station"].unique().tolist())
    fv = df[df["station"].isin(stations)]
    plt.style.use('dark_background')

    if chart == "PM2.5 Distribution by Station":
        fig, ax = plt.subplots(figsize=(10,5), facecolor='#1e2130')
        for s in stations:
            ax.hist(fv[fv["station"]==s]["PM2.5"], bins=50, alpha=0.6, label=s, density=True, color=palette.get(s,'gray'))
        ax.set_title("PM2.5 Distribution by Station", color='white')
        ax.legend()
        st.pyplot(fig, use_container_width=True)
    elif chart == "Monthly PM2.5 Trend":
        fig, ax = plt.subplots(figsize=(10,5), facecolor='#1e2130')
        m = fv.groupby(["month","station"])["PM2.5"].mean().reset_index()
        for s in stations:
            d = m[m["station"]==s]
            ax.plot(d["month"], d["PM2.5"], marker="o", label=s, color=palette.get(s,'gray'))
        ax.axhline(15, color='gray', linestyle=":", label="WHO (15)")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
    elif chart == "Hourly Pollution Pattern":
        p = st.selectbox("Pollutant:", ["PM2.5","NO2","O3","SO2","CO"])
        fig, ax = plt.subplots(figsize=(10,5), facecolor='#1e2130')
        h = fv.groupby(["hour","station"])[p].mean().reset_index()
        for s in stations:
            d = h[h["station"]==s]
            ax.plot(d["hour"], d[p], marker="o", label=s, color=palette.get(s,'gray'))
        ax.legend()
        st.pyplot(fig, use_container_width=True)
    elif chart == "PM2.5 Heatmap (Hour vs Month)":
        fig, ax = plt.subplots(figsize=(12,6), facecolor='#1e2130')
        hd = fv.groupby(["month","hour"])["PM2.5"].mean().unstack()
        sns.heatmap(hd, cmap="YlOrRd", ax=ax)
        st.pyplot(fig, use_container_width=True)
    elif chart == "Weekend vs Weekday Comparison":
        fig, ax = plt.subplots(figsize=(10,5), facecolor='#1e2130')
        wk = fv.groupby(["is_weekend","station"])["PM2.5"].mean().reset_index()
        wk["Day Type"] = wk["is_weekend"].map({0:"Weekday",1:"Weekend"})
        sns.barplot(x="station", y="PM2.5", hue="Day Type", data=wk, ax=ax)
        st.pyplot(fig, use_container_width=True)
    elif chart == "Yearly PM2.5 Trend":
        fig, ax = plt.subplots(figsize=(10,5), facecolor='#1e2130')
        yr = fv.groupby(["year","station"])["PM2.5"].mean().reset_index()
        for s in stations:
            d = yr[yr["station"]==s]
            ax.plot(d["year"], d["PM2.5"], marker="o", label=s, color=palette.get(s,'gray'))
        ax.legend()
        st.pyplot(fig, use_container_width=True)
    elif chart == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10,8), facecolor='#1e2130')
        cols = ["PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP"]
        sns.heatmap(fv[cols].corr(), annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
        st.pyplot(fig, use_container_width=True)
    elif chart == "Urban vs Suburban Comparison":
        pl = ["PM2.5","PM10","SO2","NO2","CO","O3"]
        u = df[df["station_type"]=="Urban"][pl].mean()
        s = df[df["station_type"]=="Suburban"][pl].mean()
        x = np.arange(len(pl))
        fig, ax = plt.subplots(figsize=(10,5), facecolor='#1e2130')
        ax.bar(x-0.2, u, 0.4, label="Urban", color="#ef4444")
        ax.bar(x+0.2, s, 0.4, label="Suburban", color="#22c55e")
        ax.set_xticks(x); ax.set_xticklabels(pl)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

elif page == "🤖 Model Outputs":
    st.title("🤖 PM2.5 Prediction Model")
    st.markdown("**Algorithm:** Random Forest Regressor with Lag Features")
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 R² Score", str(metrics["R2"]))
    col2.metric("📉 MAE", f"{metrics['MAE']} µg/m³")
    col3.metric("📊 RMSE", f"{metrics['RMSE']} µg/m³")
    col4.metric("🌳 Trees", "100")

    tab1, tab2, tab3 = st.tabs(["📊 Model Comparison","📉 Error Analysis","🔮 Make a Prediction"])

    with tab1:
        comp = pd.DataFrame({
            "Model": ["Linear Regression","Decision Tree","✅ Random Forest (Final)"],
            "MAE": [20.07, 14.01, metrics["MAE"]],
            "RMSE": [31.68, 26.56, metrics["RMSE"]],
            "R² Score": [0.8352, 0.8842, metrics["R2"]]
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)
        try:
            st.image("figures/model_comparison.png", use_container_width=True)
            st.image("figures/model_results.png", use_container_width=True)
        except Exception:
            st.warning("Figure images not found in figures/ folder.")

    with tab2:
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Error", "-0.01 µg/m³")
        col2.metric("Within 10 µg/m³", "80.6%")
        col3.metric("Within 20 µg/m³", "93.3%")
        try:
            st.image("figures/error_analysis.png", use_container_width=True)
        except Exception:
            st.warning("Figure image not found.")

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🏭 Pollutants")
            pm10 = st.slider("PM10", 0.0, 900.0, 80.0)
            so2 = st.slider("SO2", 0.0, 300.0, 20.0)
            no2 = st.slider("NO2", 0.0, 300.0, 50.0)
            co = st.slider("CO", 0.0, 5000.0, 800.0)
            o3 = st.slider("O3", 0.0, 400.0, 40.0)
        with col2:
            st.markdown("#### 🌤️ Weather")
            temp = st.slider("Temperature (°C)", -20.0, 42.0, 15.0)
            pres = st.slider("Pressure (hPa)", 980.0, 1050.0, 1010.0)
            dewp = st.slider("Dew Point (°C)", -40.0, 30.0, 0.0)
            rain = st.slider("Rainfall (mm)", 0.0, 60.0, 0.0)
            wspm = st.slider("Wind Speed (m/s)", 0.0, 13.0, 2.0)
        with col3:
            st.markdown("#### 📅 Time & Location")
            month = st.slider("Month", 1, 12, 6)
            hour = st.slider("Hour", 0, 23, 12)
            dow = st.slider("Day of Week", 0, 6, 2)
            is_wkd = st.selectbox("Weekend?", [0,1])
            sc = st.selectbox("Station", ["Dingling","Dongsi","Guanyuan","Huairou"])
            se = ["Dingling","Dongsi","Guanyuan","Huairou"].index(sc)
            lag1 = st.slider("PM2.5 1h ago", 0.0, 500.0, 50.0)
            lag24 = st.slider("PM2.5 24h ago", 0.0, 500.0, 50.0)

        tw = temp * wspm
        inp = np.array([[pm10,so2,no2,co,o3,temp,pres,dewp,rain,wspm,month,hour,dow,is_wkd,se,lag1,lag24,tw]])
        pred = model.predict(scaler.transform(inp))[0]

        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        col1, col2 = st.columns(2)
        col1.metric("Predicted PM2.5", f"{pred:.1f} µg/m³")
        col2.metric("WHO Guideline", "15 µg/m³",
                    delta=f"{pred-15:.1f} above limit" if pred > 15 else "Within limit",
                    delta_color="inverse")
        fig = draw_gauge(pred)
        st.pyplot(fig, use_container_width=True)
