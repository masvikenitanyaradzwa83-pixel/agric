"""
AgroVision – Climate-Smart Agricultural Advisory for Smallholder Farmers in Gweru
Simple, mobile-friendly Streamlit front-end.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from advisory_engine import AdvisoryEngine
from crop_knowledge import CROPS
from weather_analysis import climatology, annual_summary, agri_year_summary

st.set_page_config(
    page_title="AgroVision | Gweru Advisory",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def get_engine():
    return AdvisoryEngine()

eng = get_engine()

st.title("🌾 AgroVision – Climate-Smart Advisory")
st.caption("Data-driven recommendations for smallholder farmers in Gweru, Midlands, Zimbabwe")
st.markdown("---")

# Sidebar inputs
with st.sidebar:
    st.header("Your Farm Details")
    farm_size = st.number_input("Farm size (hectares)", min_value=0.1, max_value=50.0, value=1.0, step=0.1)
    soil_pH = st.slider("Soil pH (if known)", 4.0, 7.5, 5.2, 0.1)
    risk_pref = st.selectbox(
        "Risk preference",
        ["risk_averse", "neutral", "risk_seeking"],
        format_func=lambda x: {"risk_averse": "Risk-averse (prefer stable)", "neutral": "Neutral", "risk_seeking": "Risk-seeking (higher potential)"}[x],
    )
    input_access = st.selectbox(
        "Access to inputs (seed/fertilizer)",
        ["low", "medium", "high"],
        format_func=lambda x: x.capitalize(),
    )
    past_yield = st.selectbox(
        "Past season yield level",
        ["low", "medium", "high"],
        format_func=lambda x: x.capitalize(),
    )
    focus_crop = st.selectbox(
        "Crop for pest alerts",
        list(CROPS.keys()),
        format_func=lambda x: CROPS[x]["name"],
        index=0,
    )
    crop_stage = st.selectbox(
        "Current / expected crop stage",
        ["seedling", "vegetative", "whorl", "stem_elongation", "flowering", "tasseling", "podding"],
        index=1,
    )
    run = st.button("Get Advice", type="primary", use_container_width=True)

if run or "last_result" not in st.session_state:
    with st.spinner("Analysing climate data and generating recommendations..."):
        result = eng.full_advisory(
            soil_pH=soil_pH,
            farm_size_ha=farm_size,
            risk_pref=risk_pref,
            input_access=input_access,
            past_yield=past_yield,
            focus_crop=focus_crop,
            crop_stage=crop_stage,
        )
        st.session_state["last_result"] = result
else:
    result = st.session_state["last_result"]

# Layout
tab1, tab2, tab3, tab4 = st.tabs(["🌱 Crop Recommendations", "📅 Planting Window", "🐛 Pest Alerts", "📊 Climate Context"])

with tab1:
    st.subheader("Recommended Crops (ranked by suitability)")
    recs = result["crop_recommendations"]["recommendations"]
    top = result["crop_recommendations"]["top_3"]

    cols = st.columns(3)
    for i, t in enumerate(top):
        with cols[i]:
            st.metric(
                label=f"#{i+1} {t['name']}",
                value=f"{t['score']*100:.0f}%",
                help=t["notes"],
            )
            st.caption(f"Drought tol: {t['drought_tolerance']:.0%} · Inputs: {t['input_requirement']}")

    st.markdown("#### Full ranking")
    df_rec = pd.DataFrame(recs)[["name", "score", "drought_tolerance", "input_requirement", "notes"]]
    df_rec["score"] = (df_rec["score"] * 100).round(1)
    df_rec.columns = ["Crop", "Suitability %", "Drought Tolerance", "Input Need", "Notes"]
    st.dataframe(df_rec, use_container_width=True, hide_index=True)

    # Radar for top crops
    fig = go.Figure()
    for t in top[:4]:
        # Approximate axes from knowledge
        fig.add_trace(
            go.Scatterpolar(
                r=[t["score"], t["drought_tolerance"], 1.0 if t["input_requirement"] == "low" else 0.5, t["score"]],
                theta=["Overall", "Drought Tol.", "Low-input fit", "Overall"],
                fill="toself",
                name=t["name"],
            )
        )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Top crops profile")
    st.plotly_chart(fig, use_container_width=True)

    if result["crop_recommendations"]["commentary"]:
        st.info(" · ".join(result["crop_recommendations"]["commentary"]))

with tab2:
    st.subheader("Optimal Planting Window (Gweru)")
    pw = result["planting_window"]
    st.success(f"**Recommended window:** {pw.get('window_str', 'N/A')}  \n**Median onset:** {pw.get('median_date_str', 'N/A')}")
    st.write(pw.get("advice", ""))
    st.caption(f"Based on {pw.get('years_used', 0)} agricultural seasons of NASA POWER rainfall onset analysis (Oct–Feb window).")

    # Season progress
    sp = result.get("season_progress", {})
    if sp:
        st.info(f"Current agricultural season **{sp.get('agri_year')}**: {sp.get('rain_to_date_mm')} mm received so far "
                f"(data through {sp.get('last_data_date')}).")

    # Historical onsets table
    hist = pw.get("historical_onsets") or []
    if hist:
        st.markdown("#### Recent season onsets")
        hdf = pd.DataFrame(hist)
        if "onset_date" in hdf.columns:
            hdf["onset_date"] = pd.to_datetime(hdf["onset_date"]).dt.strftime("%Y-%m-%d")
        show_cols = [c for c in ["agri_year", "onset_date", "onset_month", "season_rain_mm"] if c in hdf.columns]
        hdf = hdf[show_cols].rename(columns={
            "agri_year": "Season",
            "onset_date": "Onset date",
            "onset_month": "Month",
            "season_rain_mm": "Season rain (mm)",
        })
        st.dataframe(hdf, use_container_width=True, hide_index=True)

    # Monthly rainfall bar
    clim = climatology(eng.weather)
    clim["month_name"] = clim["month"].map(
        {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
         7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    )
    clim["approx_monthly_mm"] = clim["daily_rain"] * 30.4
    fig2 = px.bar(clim, x="month_name", y="approx_monthly_mm",
                  title="Average monthly rainfall (approx, NASA POWER 2015–2024)",
                  labels={"approx_monthly_mm": "mm", "month_name": ""})
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader(f"Pest & Disease Risk – {CROPS[focus_crop]['name']} ({crop_stage})")
    alerts = result["pest_alerts"]
    if not alerts:
        st.success("No elevated risk detected for the selected crop and stage under recent weather.")
    else:
        for a in alerts:
            color = "🔴" if a["risk_level"] == "high" else "🟠"
            st.warning(f"{color} **{a['pest']}** – {a['risk_level'].upper()} risk")
            st.write(a["advice"])
            st.caption(f"Recent mean temp {a['current_mean_temp']} °C · RH {a['current_mean_rh']} %")

with tab4:
    st.subheader("Climate Context (Gweru)")
    ctx = result["crop_recommendations"]["climate_context"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Long-term avg annual rain", f"{ctx['avg_annual_rain_mm']:.0f} mm")
    c2.metric("Recent 3-yr avg rain", f"{ctx['recent_3yr_rain_mm']:.0f} mm")
    c3.metric("Mean temperature", f"{ctx['avg_temp_C']:.1f} °C")

    yearly = annual_summary(eng.weather)
    fig3 = px.line(yearly, x="year", y="total_rain_mm", markers=True,
                   title="Calendar-year rainfall (NASA POWER)", labels={"total_rain_mm": "mm"})
    st.plotly_chart(fig3, use_container_width=True)

    try:
        agri = agri_year_summary(eng.weather)
        fig4 = px.bar(agri, x="agri_year", y="total_rain_mm",
                      title="Agricultural season rainfall (Oct–Sep)", labels={"total_rain_mm": "mm", "agri_year": "Season"})
        st.plotly_chart(fig4, use_container_width=True)
    except Exception:
        pass

    st.markdown(
        """
        **Data sources**  
        - Weather: NASA POWER daily (2015–2024) for Gweru (19.45°S, 29.82°E)  
        - Crop suitability: Local agronomic optima + EcoCrop-style ranges + drought tolerance literature  
        - Model: PyTorch MLP trained on knowledge-base derived labels (Staff-ML best practice when farm outcome labels are scarce)  
        - Pest rules: Temperature/humidity phenology inspired by African pest forecasting systems (PRISE / ILCYM style)
        """
    )

st.markdown("---")
st.caption("AgroVision · Built for smallholder farmers in Gweru · Not a substitute for local extension advice")
