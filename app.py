
from pathlib import Path
from datetime import date
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sri Lankan Solar Radiation Estimator",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project folders
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
WEATHER_FILE = BASE_DIR / "weather.csv"
SOLAR_FILE = BASE_DIR / "solar.csv"

# Exact feature order used during model training
FEATURE_COLUMNS = [
    "temp_max_c",
    "temp_min_c",
    "precipitation_mm",
    "wind_speed_max_kmh",
    "humidity_mean_pct",
    "latitude",
    "longitude",
    "city",
    "month",
    "day_of_year"
]

TARGET_COLUMN = "radiation_kwh_m2"

MODEL_FILES = {
    "Random Forest": MODEL_DIR / "random_forest.pkl",
    "XGBoost": MODEL_DIR / "xgboost.pkl",
    "Stacking": MODEL_DIR / "stacking_model.pkl",
    "RF + ANN Hybrid": MODEL_DIR / "hybrid_model.pkl"
}


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    /* Google Fonts Import for ultra-clean modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Typography */
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Protect and restore Streamlit icons & sidebar collapse arrows */
    [data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="collapsedControl"] *,
    button[kind="header"] *,
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons,
    [class*="material-symbols"],
    span[translate="no"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
    }

    /* Main Page Title */
    .main-title {
        font-size: 2.85rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.03em !important;
        line-height: 1.2 !important;
    }

    .subtitle {
        color: #94a3b8 !important;
        font-size: 1.35rem !important;
        font-weight: 400 !important;
        line-height: 1.65 !important;
        margin-bottom: 2rem !important;
    }

    /* Headings Across Streamlit Containers */
    h1, [data-testid="stHeadingWithActionElements"] h1, [data-testid="stMarkdownContainer"] h1 {
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        letter-spacing: -0.025em !important;
    }

    h2, [data-testid="stHeadingWithActionElements"] h2, [data-testid="stMarkdownContainer"] h2 {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        margin-top: 2.2rem !important;
        margin-bottom: 1.1rem !important;
        letter-spacing: -0.02em !important;
    }

    h3, [data-testid="stHeadingWithActionElements"] h3, [data-testid="stMarkdownContainer"] h3 {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #e2e8f0 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
    }

    /* Paragraphs and Body Text */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div.stMarkdown p {
        font-size: 1.25rem !important;
        line-height: 1.75 !important;
        color: #cbd5e1 !important;
    }

    /* Sidebar Navigation Container */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1726 0%, #060b13 100%) !important;
        border-right: 1px solid #1e2e42 !important;
        padding-top: 1.8rem !important;
    }

    .sidebar-header {
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        margin-bottom: 0.85rem !important;
        padding-left: 2px !important;
    }

    /* Sidebar Dropdown Container */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #132235 !important;
        border: 2px solid #294263 !important;
        border-radius: 14px !important;
        min-height: 56px !important;
        padding: 4px 10px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"]:hover {
        border-color: #38bdf8 !important;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.25) !important;
    }

    /* Sidebar Dropdown Selected Text */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] div,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] span {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }

    /* Preserve the dropdown arrow icon */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] svg {
        width: 26px !important;
        height: 26px !important;
        fill: #38bdf8 !important;
    }

    /* Dropdown Options Popup Menu (BaseWeb Portal) */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div {
        background-color: #0d1928 !important;
        border: 2px solid #294263 !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6) !important;
    }

    ul[role="listbox"],
    ul[role="listbox"] *,
    li[role="option"],
    li[role="option"] *,
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"] * {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }

    ul[role="listbox"] li,
    li[role="option"] {
        padding: 16px 20px !important;
        margin: 4px 0 !important;
        border-radius: 10px !important;
        cursor: pointer !important;
        color: #e2e8f0 !important;
    }

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #1e3552 !important;
        color: #38bdf8 !important;
    }

    /* Metric Cards - Sleek Dark Slate with High Contrast */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #182334 0%, #0f172a 100%) !important;
        border: 1px solid #2d3f56 !important;
        padding: 24px 28px !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.5) !important;
        border-color: #38bdf8 !important;
    }

    /* Metric Label (Upper text inside card) */
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
        color: #94a3b8 !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }

    /* Metric Value (Main number/text inside card) */
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: #38bdf8 !important;
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        line-height: 1.2 !important;
    }

    /* Info Note Box */
    .info-note {
        background: rgba(14, 165, 233, 0.08) !important;
        border: 1px solid rgba(14, 165, 233, 0.35) !important;
        border-left: 6px solid #0ea5e9 !important;
        padding: 20px 28px !important;
        border-radius: 14px !important;
        color: #e0f2fe !important;
        font-size: 1.25rem !important;
        line-height: 1.75 !important;
        margin-bottom: 2rem !important;
    }

    /* Alert boxes (st.success, st.info, st.warning) */
    div[data-testid="stAlert"],
    div[data-testid="stAlert"] * {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        line-height: 1.6 !important;
    }

    div[data-testid="stAlert"] {
        padding: 18px 24px !important;
        border-radius: 12px !important;
        margin: 1.4rem 0 !important;
    }

    /* HTML Tables (st.table) - Large, clean, high contrast */
    div[data-testid="stTable"] {
        margin: 1.6rem 0 !important;
        overflow-x: auto !important;
    }

    div[data-testid="stTable"] table {
        width: 100% !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border: 2px solid #2d4361 !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        background-color: #101c2d !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35) !important;
    }

    div[data-testid="stTable"] th {
        background-color: #16263b !important;
        color: #38bdf8 !important;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        padding: 18px 24px !important;
        text-align: left !important;
        border-bottom: 2px solid #2d4361 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }

    div[data-testid="stTable"] td {
        background-color: #0d1726 !important;
        color: #f8fafc !important;
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        padding: 18px 24px !important;
        border-bottom: 1px solid #1c2e44 !important;
    }

    div[data-testid="stTable"] tr:last-child td {
        border-bottom: none !important;
    }

    div[data-testid="stTable"] tr:hover td {
        background-color: #16283e !important;
        color: #38bdf8 !important;
    }

    /* Form labels, inputs & buttons */
    .stSelectbox label, .stDateInput label, .stNumberInput label {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
        margin-bottom: 0.5rem !important;
    }

    .stSelectbox div[data-baseweb="select"] *,
    .stNumberInput input,
    .stDateInput input {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }

    .stSelectbox div[data-baseweb="select"] {
        min-height: 54px !important;
        border-radius: 12px !important;
        background-color: #121f31 !important;
        border: 2px solid #2d4463 !important;
    }

    .stSlider label,
    .stSlider [data-testid="stWidgetLabel"] p {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
        margin-bottom: 0.6rem !important;
    }

    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.35rem !important;
        padding: 0.9rem 2rem !important;
    }

    div[data-testid="stDataFrame"] {
        font-size: 1.25rem !important;
        border-radius: 12px !important;
    }

    /* Paragraphs and list items */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div.stMarkdown p,
    div.stMarkdown li {
        font-size: 1.3rem !important;
        line-height: 1.8 !important;
        color: #cbd5e1 !important;
    }

    div[data-testid="stMarkdownContainer"] strong,
    div.stMarkdown strong {
        color: #ffffff !important;
        font-weight: 750 !important;
    }

    /* Footer Text */
    .footer-text {
        color: #64748b !important;
        font-size: 1.15rem !important;
        text-align: center !important;
        padding: 1.2rem 0 2rem 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. LOAD SAVED MODELS AND DATA
# ============================================================

@st.cache_resource
def load_models():
    """Load the four previously trained models."""

    loaded = {}

    for model_name, model_path in MODEL_FILES.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        loaded[model_name] = joblib.load(model_path)

    return loaded


@st.cache_data
def load_datasets():
    """Load the original weather and solar datasets."""

    if not WEATHER_FILE.exists():
        raise FileNotFoundError(
            f"Weather dataset not found: {WEATHER_FILE}"
        )

    if not SOLAR_FILE.exists():
        raise FileNotFoundError(
            f"Solar dataset not found: {SOLAR_FILE}"
        )

    weather = pd.read_csv(WEATHER_FILE)
    solar = pd.read_csv(SOLAR_FILE)

    return weather, solar


@st.cache_data
def load_results():
    """Load saved model metrics and test predictions."""

    metrics_path = RESULT_DIR / "model_metrics.csv"
    predictions_path = RESULT_DIR / "test_predictions.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}"
        )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Test predictions file not found: {predictions_path}"
        )

    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)

    return metrics, predictions


# ============================================================
# 4. GENERAL HELPER FUNCTIONS
# ============================================================

def get_city_coordinates(weather_data):
    """
    Get one latitude/longitude pair for each city.

    The original weather dataset should contain:
    city, latitude and longitude.
    """

    required = {"city", "latitude", "longitude"}

    if not required.issubset(weather_data.columns):
        missing = required - set(weather_data.columns)

        raise ValueError(
            "The weather dataset is missing coordinate columns: "
            + ", ".join(missing)
        )

    locations = weather_data[
        ["city", "latitude", "longitude"]
    ].dropna().copy()

    locations["city"] = locations["city"].astype(str).str.strip()

    locations = (
        locations.groupby("city", as_index=False)
        .agg({
            "latitude": "median",
            "longitude": "median"
        })
        .sort_values("city")
    )

    return locations


def get_stacking_prediction(stacking_data, input_data):
    """
    Generate stacking predictions.

    The base prediction order must match the order used
    when fitting the Ridge meta-model in the notebook:
    Random Forest, XGBoost, ANN.
    """

    base_models = stacking_data["base_models"]
    meta_model = stacking_data["meta_model"]

    rf_prediction = base_models["random_forest"].predict(input_data)
    xgb_prediction = base_models["xgboost"].predict(input_data)
    ann_prediction = base_models["mlp"].predict(input_data)

    meta_features = np.column_stack([
        rf_prediction,
        xgb_prediction,
        ann_prediction
    ])

    return meta_model.predict(meta_features)


def get_all_predictions(models, input_data):
    """Generate predictions from all four trained approaches."""

    predictions = {}

    predictions["Random Forest"] = float(
        models["Random Forest"].predict(input_data)[0]
    )

    predictions["XGBoost"] = float(
        models["XGBoost"].predict(input_data)[0]
    )

    predictions["Stacking"] = float(
        get_stacking_prediction(
            models["Stacking"],
            input_data
        )[0]
    )

    hybrid_data = models["RF + ANN Hybrid"]

    rf_prediction = hybrid_data["random_forest"].predict(input_data)[0]
    ann_prediction = hybrid_data["ann"].predict(input_data)[0]

    hybrid_prediction = (
        hybrid_data["rf_weight"] * rf_prediction
        + hybrid_data["ann_weight"] * ann_prediction
    )

    predictions["RF + ANN Hybrid"] = float(hybrid_prediction)

    return predictions


def get_best_model(metrics):
    """Find the model with the lowest MAE."""

    if "MAE" not in metrics.columns:
        return None

    if "Model" not in metrics.columns:
        return None

    valid_metrics = metrics.dropna(subset=["MAE"])

    if valid_metrics.empty:
        return None

    best_row = valid_metrics.loc[valid_metrics["MAE"].idxmin()]

    return best_row


def show_page_header(title, description):
    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="subtitle">{description}</div>',
        unsafe_allow_html=True
    )


# ============================================================
# 5. LOAD PROJECT FILES
# ============================================================

try:
    models = load_models()
    weather_data, solar_data = load_datasets()
    metrics_df, test_results = load_results()

    city_locations = get_city_coordinates(weather_data)

except Exception as error:
    st.error("The dashboard could not load the required project files.")
    st.exception(error)

    st.info(
        "Check that the models, results and CSV files are in "
        "the folders shown in the project structure."
    )

    st.stop()


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown('<div class="sidebar-header">Navigation</div>', unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Overview",
        "Solar Radiation Prediction",
        "Ensemble Model Comparison",
        "Model Insights"
    ],
    label_visibility="collapsed"
)


# ============================================================
# 7. PAGE 1 - OVERVIEW
# ============================================================

if page == "Overview":

    show_page_header(
        "Sri Lankan Solar Radiation Estimator",
        "An ensemble learning dashboard for estimating daily solar radiation "
        "using weather and geographical information."
    )

    st.markdown(
        """
        <div class="info-note">
        This project uses machine learning ensemble techniques to estimate
        daily solar radiation from weather conditions and location.
        The results can support preliminary solar-energy assessment and
        help identify locations and periods that may be suitable for
        solar power planning.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## Project at a glance")

    total_records = len(weather_data)
    total_cities = weather_data["city"].nunique()
    total_models = len(models)

    best_model = get_best_model(metrics_df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Weather Records", f"{total_records:,}")
    col2.metric("Sri Lankan Cities", f"{total_cities:,}")
    col3.metric("Ensemble Approaches", total_models)

    if best_model is not None:
        col4.metric("Best Model", str(best_model["Model"]))
    else:
        col4.metric("Best Model", "Not available")

    st.markdown("## Dataset coverage")

    if "date" in weather_data.columns:
        weather_data["date"] = pd.to_datetime(
            weather_data["date"],
            errors="coerce"
        )

        date_col1, date_col2 = st.columns(2)

        date_col1.metric(
            "First Available Date",
            str(weather_data["date"].min().date())
        )

        date_col2.metric(
            "Last Available Date",
            str(weather_data["date"].max().date())
        )


# ============================================================
# 8. PAGE 2 - SOLAR RADIATION PREDICTION
# ============================================================

elif page == "Solar Radiation Prediction":

    show_page_header(
        "Solar Radiation Prediction",
        "Enter daily weather conditions to estimate solar radiation "
        "using the four trained machine learning models."
    )

    st.info(
        "This page estimates solar radiation for the selected date using "
        "weather conditions entered by the user. It does not forecast "
        "future weather conditions."
    )

    with st.form("prediction_form"):

        st.markdown("### 1. Select location and date")

        col1, col2 = st.columns(2)

        city_list = city_locations["city"].tolist()

        with col1:
            selected_city = st.selectbox(
                "Select a Sri Lankan city",
                city_list
            )

        with col2:
            selected_date = st.date_input(
                "Select date",
                value=date.today()
            )

        st.markdown("### 2. Enter weather conditions")

        col1, col2 = st.columns(2)

        with col1:
            temp_max = st.number_input(
                "Maximum temperature (°C)",
                min_value=-10.0,
                max_value=60.0,
                value=30.0,
                step=0.5
            )

            temp_min = st.number_input(
                "Minimum temperature (°C)",
                min_value=-10.0,
                max_value=60.0,
                value=24.0,
                step=0.5
            )

            precipitation = st.number_input(
                "Precipitation (mm)",
                min_value=0.0,
                max_value=1000.0,
                value=0.0,
                step=0.5
            )

        with col2:
            wind_speed = st.number_input(
                "Maximum wind speed (km/h)",
                min_value=0.0,
                max_value=300.0,
                value=15.0,
                step=0.5
            )

            humidity = st.number_input(
                "Mean humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0
            )

        submitted = st.form_submit_button(
            "☀️ Estimate Solar Radiation",
            use_container_width=True
        )

    if submitted:

        if temp_min > temp_max:
            st.error(
                "Minimum temperature cannot be greater than "
                "maximum temperature."
            )

        else:

            selected_location = city_locations[
                city_locations["city"] == selected_city
            ].iloc[0]

            latitude = float(selected_location["latitude"])
            longitude = float(selected_location["longitude"])

            input_row = pd.DataFrame([{
                "temp_max_c": temp_max,
                "temp_min_c": temp_min,
                "precipitation_mm": precipitation,
                "wind_speed_max_kmh": wind_speed,
                "humidity_mean_pct": humidity,
                "latitude": latitude,
                "longitude": longitude,
                "city": selected_city,
                "month": selected_date.month,
                "day_of_year": selected_date.timetuple().tm_yday
            }])

            input_row = input_row[FEATURE_COLUMNS]

            try:
                prediction_results = get_all_predictions(
                    models,
                    input_row
                )

                st.session_state["latest_predictions"] = prediction_results
                st.session_state["latest_input"] = input_row

            except Exception as error:
                st.error("Prediction failed.")
                st.exception(error)

    if "latest_predictions" in st.session_state:

        prediction_results = st.session_state["latest_predictions"]

        st.markdown("---")
        st.markdown("## Estimated solar radiation")

        best_prediction_model = min(
            prediction_results,
            key=prediction_results.get
        )

        # The lowest predicted value is not necessarily the best model.
        # Show all predictions without declaring one the most accurate.
        prediction_columns = st.columns(4)

        for column, (model_name, prediction) in zip(
            prediction_columns,
            prediction_results.items()
        ):
            column.metric(
                model_name,
                f"{prediction:.3f} kWh/m²/day"
            )

        prediction_df = pd.DataFrame({
            "Model": list(prediction_results.keys()),
            "Estimated Solar Radiation (kWh/m²/day)": list(
                prediction_results.values()
            )
        })

        fig = px.bar(
            prediction_df,
            x="Model",
            y="Estimated Solar Radiation (kWh/m²/day)",
            color="Model",
            title="Predictions from the Four Models"
        )

        fig.update_traces(
            texttemplate="%{y:.3f}",
            textposition="outside",
            textfont=dict(size=20, color="#ffffff", family="Plus Jakarta Sans"),
            cliponaxis=False
        )

        fig.update_layout(
            showlegend=False,
            title=dict(
                text="Predictions from the Four Models",
                font=dict(size=24, color="#f8fafc", family="Plus Jakarta Sans")
            ),
            xaxis=dict(
                title=dict(text="Model", font=dict(size=20, color="#f8fafc", family="Plus Jakarta Sans")),
                tickfont=dict(size=18, color="#cbd5e1", family="Plus Jakarta Sans")
            ),
            yaxis=dict(
                title=dict(text="Solar Radiation (kWh/m²/day)", font=dict(size=20, color="#f8fafc", family="Plus Jakarta Sans")),
                tickfont=dict(size=18, color="#cbd5e1", family="Plus Jakarta Sans")
            ),
            margin=dict(t=70, b=50, l=60, r=40),
            height=540
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Input information")

        st.table(
            st.session_state["latest_input"]
        )

        st.caption(
            "These are model estimates, not measured solar radiation "
            "values. Prediction accuracy depends on the quality and "
            "representativeness of the supplied weather conditions."
        )

        download_df = prediction_df.copy()

        csv_data = download_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download prediction results",
            data=csv_data,
            file_name="solar_radiation_predictions.csv",
            mime="text/csv"
        )


# ============================================================
# 9. PAGE 3 - ENSEMBLE MODEL COMPARISON
# ============================================================

elif page == "Ensemble Model Comparison":

    show_page_header(
        "Ensemble Model Comparison",
        "Compare the performance of Bagging, Boosting, Stacking "
        "and Hybridization using the saved test results."
    )

    required_metric_columns = {
        "Model", "MAE", "RMSE", "R2 Score"
    }

    if not required_metric_columns.issubset(metrics_df.columns):

        st.error(
            "The model_metrics.csv file does not contain all expected "
            "columns: Model, MAE, RMSE and R2 Score."
        )

    else:

        best_model_row = get_best_model(metrics_df)

        if best_model_row is not None:
            st.success(
                f"Best model based on MAE: {best_model_row['Model']} "
                f"(MAE: {best_model_row['MAE']:.4f})"
            )

        st.markdown("## Performance metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Best MAE",
            f"{metrics_df['MAE'].min():.4f}"
        )

        col2.metric(
            "Best RMSE",
            f"{metrics_df['RMSE'].min():.4f}"
        )

        col3.metric(
            "Best R² Score",
            f"{metrics_df['R2 Score'].max():.4f}"
        )

        st.markdown("## Complete model comparison")

        display_metrics = metrics_df.round(4).copy()
        if "Model" in display_metrics.columns:
            display_metrics = display_metrics.set_index("Model")

        st.table(display_metrics)

        st.markdown("## Compare evaluation metrics")

        selected_metric = st.selectbox(
            "Select evaluation metric",
            ["MAE", "RMSE", "R2 Score"]
        )

        fig = px.bar(
            metrics_df,
            x="Model",
            y=selected_metric,
            color="Model",
            text_auto=".4f",
            title=f"Model Comparison - {selected_metric}"
        )

        fig.update_traces(
            textfont=dict(size=20, color="#ffffff", family="Plus Jakarta Sans"),
            textposition="outside",
            cliponaxis=False
        )

        fig.update_layout(
            showlegend=False,
            title=dict(
                text=f"Model Comparison - {selected_metric}",
                font=dict(size=24, color="#f8fafc", family="Plus Jakarta Sans")
            ),
            xaxis=dict(
                title=dict(text="Model", font=dict(size=20, color="#f8fafc", family="Plus Jakarta Sans")),
                tickfont=dict(size=18, color="#cbd5e1", family="Plus Jakarta Sans")
            ),
            yaxis=dict(
                title=dict(text=selected_metric, font=dict(size=20, color="#f8fafc", family="Plus Jakarta Sans")),
                tickfont=dict(size=18, color="#cbd5e1", family="Plus Jakarta Sans")
            ),
            margin=dict(t=70, b=50, l=60, r=40),
            height=540
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## Actual vs predicted values")

        prediction_columns = {
            "random_forest_prediction": "Random Forest",
            "xgboost_prediction": "XGBoost",
            "stacking_prediction": "Stacking",
            "hybrid_prediction": "RF + ANN Hybrid"
        }

        required_prediction_columns = {
            "actual_radiation",
            *prediction_columns.keys()
        }

        if required_prediction_columns.issubset(test_results.columns):

            selected_model = st.selectbox(
                "Select model for actual vs predicted comparison",
                list(prediction_columns.values())
            )

            selected_prediction_column = next(
                column
                for column, name in prediction_columns.items()
                if name == selected_model
            )

            plot_data = test_results[
                ["actual_radiation", selected_prediction_column]
            ].dropna()

            # Use a sample for a responsive interactive chart
            if len(plot_data) > 3000:
                plot_data = plot_data.sample(
                    n=3000,
                    random_state=42
                )

            fig = px.scatter(
                plot_data,
                x="actual_radiation",
                y=selected_prediction_column,
                opacity=0.55,
                title=f"{selected_model}: Actual vs Predicted"
            )

            min_value = min(
                plot_data["actual_radiation"].min(),
                plot_data[selected_prediction_column].min()
            )

            max_value = max(
                plot_data["actual_radiation"].max(),
                plot_data[selected_prediction_column].max()
            )

            fig.add_trace(
                go.Scatter(
                    x=[min_value, max_value],
                    y=[min_value, max_value],
                    mode="lines",
                    name="Perfect prediction",
                    line=dict(dash="dash", width=3, color="#f59e0b")
                )
            )

            fig.update_layout(
                title=dict(
                    text=f"{selected_model}: Actual vs Predicted",
                    font=dict(size=24, color="#f8fafc", family="Plus Jakarta Sans")
                ),
                xaxis=dict(
                    title=dict(text="Actual Solar Radiation (kWh/m²/day)", font=dict(size=20, color="#f8fafc", family="Plus Jakarta Sans")),
                    tickfont=dict(size=18, color="#cbd5e1", family="Plus Jakarta Sans")
                ),
                yaxis=dict(
                    title=dict(text="Predicted Solar Radiation (kWh/m²/day)", font=dict(size=20, color="#f8fafc", family="Plus Jakarta Sans")),
                    tickfont=dict(size=18, color="#cbd5e1", family="Plus Jakarta Sans")
                ),
                legend=dict(
                    font=dict(size=18, color="#e2e8f0", family="Plus Jakarta Sans")
                ),
                margin=dict(t=70, b=50, l=60, r=40),
                height=560
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(
                "The saved test_predictions.csv file is missing "
                "one or more expected prediction columns."
            )

        st.markdown("## Model performance interpretation")

        st.write(
            """
            - **MAE:** Measures the average absolute difference between
              actual and predicted solar radiation. Lower is better.
            - **RMSE:** Penalizes larger prediction errors more heavily.
              Lower is better.
            - **R²:** Measures how much of the variation in the target
              is explained by the model. Higher is generally better.
            """
        )


# ============================================================
# 10. PAGE 4 - MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    show_page_header(
        "Model Insights",
        "Explore the features used by the models and understand "
        "their contribution to solar radiation estimation."
    )

    st.markdown("## Random Forest feature importance")

    rf_pipeline = models["Random Forest"]

    try:
        preprocessor = rf_pipeline.named_steps["preprocessor"]
        rf_estimator = rf_pipeline.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()
        feature_importances = rf_estimator.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importances
        }).sort_values(
            "Importance",
            ascending=False
        )

        top_n = st.slider(
            "Number of features to display",
            min_value=5,
            max_value=min(35, len(importance_df)),
            value=min(10, len(importance_df))
        )

        top_features = importance_df.head(top_n)

        fig = px.bar(
            top_features.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance"
        )

        fig.update_layout(
            title=dict(
                text="Random Forest Feature Importance",
                font=dict(size=24, color="#f8fafc", family="Plus Jakarta Sans")
            ),
            xaxis=dict(
                title=dict(text="Feature Importance", font=dict(size=20, color="#f8fafc", family="Plus Jakarta Sans")),
                tickfont=dict(size=18, color="#cbd5e1", family="Plus Jakarta Sans")
            ),
            yaxis=dict(
                title=dict(text="Feature", font=dict(size=20, color="#f8fafc", family="Plus Jakarta Sans")),
                tickfont=dict(size=18, color="#cbd5e1", family="Plus Jakarta Sans")
            ),
            margin=dict(t=70, b=50, l=120, r=40),
            height=550
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Feature importance values")

        table_df = top_features.copy()
        table_df["Importance"] = table_df["Importance"].map(lambda x: f"{x:.5f}")
        table_df = table_df.set_index("Feature")

        st.table(table_df)

        st.info(
            "These values are Random Forest built-in feature importances."
        )

    except Exception as error:
        st.warning(
            "Could not extract Random Forest feature importances."
        )
        st.exception(error)


# ============================================================
# 11. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer-text">Sri Lankan Solar Radiation Prediction Using Ensemble Learning</div>',
    unsafe_allow_html=True
)
