import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import base64
import plotly.express as px
import streamlit.components.v1 as components

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.classification import (
    LogisticRegressionModel,
    DecisionTreeClassificationModel,
    RandomForestClassificationModel,
)

import os
import sys
import streamlit as st
from pyspark.sql import SparkSession


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

RAILWAY_BANNER = os.path.join(ASSETS_DIR, "railway_banner.png")
ANALYTICS_BG = os.path.join(ASSETS_DIR, "analytics_background.png")
DELAY_CONCEPT = os.path.join(ASSETS_DIR, "delay_concept.png")
PREDICTION_WORKFLOW = os.path.join(ASSETS_DIR, "prediction_workflow.png")

import streamlit.components.v1 as components
def scroll_to_top():
    components.html(
        """
        <script>
        function resetParentScroll() {
            try {
                const doc = window.parent.document;

                const selectors = [
                    '[data-testid="stAppViewContainer"]',
                    '[data-testid="stMain"]',
                    'section.main'
                ];

                selectors.forEach(selector => {
                    const element = doc.querySelector(selector);
                    if (element) {
                        element.scrollTo({
                            top: 0,
                            left: 0,
                            behavior: "instant"
                        });
                    }
                });

                window.parent.scrollTo(0, 0);
            } catch (error) {
                console.log("Scroll reset unavailable", error);
            }
        }

        resetParentScroll();
        setTimeout(resetParentScroll, 100);
        setTimeout(resetParentScroll, 300);
        setTimeout(resetParentScroll, 700);
        </script>
        """,
        height=1,
    )



# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Railway Delay Prediction Using Apache Spark",
    page_icon="🚆",
    layout="wide",
)


st.markdown("""
<style>
/* Keep text readable over the dark dashboard background */
.stApp,
.stApp p,
.stApp label,
.stApp [data-testid="stMarkdownContainer"] p {
    color: #F1F5F9;
}

/* Headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #FFFFFF !important;
}

/* Captions and secondary text */
.stApp [data-testid="stCaptionContainer"] {
    color: #CBD5E1 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(20, 24, 38, 0.88);
    border: 1px solid rgba(255, 255, 255, 0.20);
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.18);
}

[data-testid="stMetric"] label,
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #FFFFFF !important;
}

/* Dataframes and tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    color: #F1F5F9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    color: #F1F5F9;
}
</style>
""", unsafe_allow_html=True)



# --------------------------------------------------
# DASHBOARD BACKGROUND
# --------------------------------------------------

import base64
import streamlit as st
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"

def set_page_background(image_name):
    image_path = ASSETS_DIR / image_name

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(
                    rgba(10, 15, 25, 0.78),
                    rgba(10, 15, 25, 0.86)
                ),
                url("data:image/png;base64,{encoded_image}");

            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )




BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

DATA_URL = (
    "https://huggingface.co/datasets/"
    "abhiramvinjamuri/railway-delay-data/resolve/main/ir_train.csv"
)

# --------------------------------------------------
# LOAD SAVED FILES
# --------------------------------------------------

st.title("Railway Delay Prediction Using Apache Spark")


st.caption(
    "Big Data Ecosystem Case Study | PySpark · Spark MLlib · Streamlit"
)



if not DATA_URL:
    st.error("Dataset URL is not configured.")
    st.stop()

required_paths = [
    MODEL_DIR / "preprocessor",
    MODEL_DIR / "logistic_regression",
    MODEL_DIR / "decision_tree",
    MODEL_DIR / "random_forest",
    MODEL_DIR / "metrics.json",
    MODEL_DIR / "metadata.json",
]

missing = [str(p) for p in required_paths if not p.exists()]

if missing:
    st.error("Some exported model files are missing:")
    for item in missing:
        st.write(f"- {item}")
    st.stop()

with open(MODEL_DIR / "metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)

with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)



# Import PySpark only after configuring the environment
from pyspark.sql import SparkSession


# --------------------------------------------------
# START SPARK
# --------------------------------------------------

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
from pyspark.sql import SparkSession

@st.cache_resource
def get_spark():
    import os
    import sys
    import shutil
    from pyspark.sql import SparkSession

    java = shutil.which("java")
    if not java:
        raise RuntimeError("Java was not found in PATH.")

    os.environ["JAVA_HOME"] = os.path.dirname(
        os.path.dirname(os.path.realpath(java))
    )
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder
        .appName("RailwayDelayRiskPrediction")
        .master("local[*]")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )

    return spark

# --------------------------------------------------
# LOADING SCREEN
# --------------------------------------------------


def show_loading_screen():
    placeholder = st.empty()

    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">

    <style>
        * {
            box-sizing: border-box;
        }

        html, body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            background: #0e0e13;
            font-family: Arial, sans-serif;
            color: white;
            overflow: hidden;
        }

        .loading-screen {
            min-height: 100vh;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 12px;
        }

        .loading-card {
            width: 100%;
            max-width: 600px;
            padding: 38px 30px;
            text-align: center;
            border: 1px solid #343442;
            border-radius: 18px;
            background: #13131c;
        }

        /* Train icon */
        .icon {
            width: 66px;
            height: 66px;
            margin: 0 auto 20px;
            border: 2px solid #555568;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 31px;
            color: #ffffff;
            animation: trainPulse 2s ease-in-out infinite;
        }

        @keyframes trainPulse {
            0%, 100% {
                transform: scale(1);
                border-color: #555568;
            }
            50% {
                transform: scale(1.06);
                border-color: #a6a6b8;
            }
        }

        h2 {
            margin: 0 0 12px;
            font-size: 27px;
            font-weight: 600;
            color: #ffffff;
        }

        .subtitle {
            color: #a6a6b8;
            font-size: 14px;
            line-height: 1.6;
            margin: 0 auto 28px;
        }

        .spinner {
            width: 38px;
            height: 38px;
            margin: 0 auto 22px;
            border: 3px solid #343442;
            border-top: 3px solid #ffffff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }

        .bar {
            width: 100%;
            height: 5px;
            background: #302f3b;
            border-radius: 10px;
            overflow: hidden;
        }

        .bar-fill {
            width: 35%;
            height: 100%;
            background: #d6d6e3;
            border-radius: 10px;
            animation: progress 1.8s ease-in-out infinite;
        }

        @keyframes progress {
            0% {
                transform: translateX(-110%);
            }
            100% {
                transform: translateX(300%);
            }
        }

        .status {
            margin: 18px 0 0;
            font-size: 13px;
            color: #a6a6b8;
        }

        .footer {
            margin-top: 26px;
            font-size: 11px;
            color: #686879;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }
    </style>
    </head>

    <body>
        <div class="loading-screen">
            <div class="loading-card">

                <div class="icon">&#128646;</div>

                <h2>Railway Delay Prediction</h2>

                <p class="subtitle">
                    Initializing the prediction system.
                    Please wait while the required components load.
                </p>

                <div class="spinner"></div>

                <div class="bar">
                    <div class="bar-fill"></div>
                </div>

                <p class="status">
                    Initializing Spark · Loading dataset · Preparing models
                </p>

                <div class="footer">
                    Big Data Ecosystem Project
                </div>

            </div>
        </div>
    </body>
    </html>
    """

    with placeholder:
        components.html(html, height=650, scrolling=False)

    return placeholder


import tempfile
import urllib.request

@st.cache_resource
def download_dataset():
    dataset_path = os.path.join(
        tempfile.gettempdir(),
        "ir_train.csv"
    )

    if not os.path.exists(dataset_path):
        urllib.request.urlretrieve(DATA_URL, dataset_path)

    return dataset_path


@st.cache_resource
def load_data():
    spark_session = get_spark()
    dataset_path = download_dataset()

    return (
        spark_session.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(dataset_path)
    )

# --------------------------------------------------
# LOAD DATA AND SAVED MODELS
# --------------------------------------------------

spark = get_spark()
df = load_data()

preprocessor = PipelineModel.load(
    str(MODEL_DIR / "preprocessor")
)

models = {
    "Logistic Regression": LogisticRegressionModel.load(
        str(MODEL_DIR / "logistic_regression")
    ),
    "Decision Tree": DecisionTreeClassificationModel.load(
        str(MODEL_DIR / "decision_tree")
    ),
    "Random Forest": RandomForestClassificationModel.load(
        str(MODEL_DIR / "random_forest")
    ),
}




# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

pages = [
    "Introduction",
    "Overview",
    "Big Data Technologies",
    "Model Comparison",
    "Model Evaluation",
    "Delay Insights",
    "Train Delay Predictor",
]


def navigate_to_page(destination):
    st.session_state["page_navigation"] = destination

page = st.sidebar.radio(
    "Go to",
    pages,
    key="page_navigation",
)

page_changed = st.session_state.get("previous_page") != page
st.session_state["previous_page"] = page



# --------------------------------------------------
# INTRODUCTION PAGE
# --------------------------------------------------

if page == "Introduction":
    set_page_background("railway_banner.png")

    st.subheader("Project Overview")

    st.markdown(
        """
        This project uses **Big Data and Machine Learning** to predict
        whether a train's arrival delay will exceed 15 minutes.

        Apache Spark and PySpark MLlib are used to process the railway
        dataset and train classification models. The Streamlit dashboard
        presents journey statistics, model performance, delay patterns,
        and interactive predictions.
        """
    )

    st.divider()

    st.header("Problem Statement")

    st.write(
        """
        Train delays affect passengers, railway operations, and
        connecting services. The objective is to predict whether a train
        will arrive more than 15 minutes late using journey, route,
        schedule, environmental, and operational features.
        """
    )

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", "15 Lakhs")
    col2.metric("Dataset Columns", "45")
    col3.metric("Task", "Binary Classification")

    st.markdown("**Target Variable: `is_delayed`**")

    st.dataframe(
        pd.DataFrame({
            "Value": [0, 1],
            "Meaning": [
                "On time or delayed by 15 minutes or less",
                "Arrival delay greater than 15 minutes",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The actual delay duration and recorded delay cause are excluded "
        "from model inputs to prevent data leakage."
    )

    st.divider()

    st.header("Dataset Variables")

    variable_descriptions = {
        "journey_id": "Unique identifier for a train journey.",
        "train_number": "Number identifying the train service.",
        "train_type": "Category or type of train service.",
        "departure_date": "Scheduled journey departure date.",
        "year": "Year of the journey.",
        "month": "Month of the journey.",
        "day_of_week": "Day of the week for the journey.",
        "departure_hour": "Hour at which the train departs.",
        "is_weekend": "Indicates whether the journey is on a weekend.",
        "is_night_departure": "Indicates a night-time departure.",
        "is_peak_hour": "Indicates departure during peak hours.",
        "is_festival_season": "Indicates a festival-season period.",
        "season": "Season associated with the journey.",
        "zone": "Railway zone associated with the journey.",
        "zone_abbr": "Abbreviation of the railway zone.",
        "source_station_category": "Category of the origin station.",
        "destination_station_category": "Category of the destination station.",
        "distance_km": "Journey distance in kilometres.",
        "num_scheduled_stops": "Number of scheduled stops.",
        "scheduled_travel_hours": "Scheduled journey duration in hours.",
        "track_doubled": "Indicates whether the route has doubled track.",
        "is_hdn_route": "Indicates whether the route is an HDN route.",
        "is_electrified": "Indicates whether the route is electrified.",
        "psr_count": "Number of permanent speed restrictions.",
        "is_circular_route": "Indicates whether the route is circular.",
        "is_monsoon_season": "Indicates a monsoon-season journey.",
        "is_fog_risk": "Indicates whether fog risk is present.",
        "fog_risk_score": "Numerical score representing fog risk.",
        "zone_fog_index": "Fog-related index for the railway zone.",
        "zone_congestion_index": "Congestion index for the railway zone.",
        "season_severity_score": "Score representing seasonal severity.",
        "traction_type": "Type of train traction or propulsion.",
        "loco_age_years": "Age of the locomotive in years.",
        "coach_age_years": "Age of the coaches in years.",
        "has_lhb_coaches": "Indicates whether LHB coaches are used.",
        "is_rake_shared": "Indicates whether the rake is shared.",
        "maintenance_score": "Score representing maintenance condition.",
        "seat_utilisation_pct": "Percentage of seat utilisation.",
        "is_overloaded": "Indicates whether the train is overloaded.",
        "late_incoming_rake": "Indicates whether the incoming rake is late.",
        "is_special_train": "Indicates whether the service is a special train.",
        "route_historical_ontime_pct": "Historical on-time percentage for the route.",
        "delay_minutes": "Actual arrival delay; excluded from model inputs.",
        "primary_delay_cause": "Recorded delay cause; excluded from model inputs.",
        "is_delayed": "Target: 1 if delay exceeds 15 minutes, otherwise 0.",
    }

    categorical_variables = {
        "train_type", "season", "zone", "zone_abbr",
        "source_station_category", "destination_station_category",
        "traction_type", "journey_id", "departure_date",
        "primary_delay_cause",
    }

    binary_variables = {
        "is_weekend", "is_night_departure", "is_peak_hour",
        "is_festival_season", "track_doubled", "is_hdn_route",
        "is_electrified", "is_circular_route", "is_monsoon_season",
        "is_fog_risk", "has_lhb_coaches", "is_rake_shared",
        "is_overloaded", "late_incoming_rake", "is_special_train",
        "is_delayed",
    }

    excluded_variables = {
        "journey_id", "departure_date",
        "delay_minutes", "primary_delay_cause",
    }

    variable_rows = []

    for sno, column in enumerate(df.columns, start=1):
        if column in excluded_variables:
            role = "Identifier / date / leakage"
        elif column == "is_delayed":
            role = "Target variable (binary)"
        elif column in binary_variables:
            role = "Binary / numerical flag"
        elif column in categorical_variables:
            role = "Categorical"
        else:
            role = "Numerical"

        variable_rows.append({
            "S.No.": sno,
            "Variable": column,
            "Data Type": str(df.schema[column].dataType),
            "Category": role,
            "Description": variable_descriptions.get(
                column, "Description to be verified against the dataset."
            ),
        })

    variable_table = pd.DataFrame(variable_rows)

    search_variable = st.text_input(
        "Search dataset variables",
        placeholder="Search by variable name or description",
    )

    if search_variable:
        variable_table = variable_table[
            variable_table.apply(
                lambda row: row.astype(str).str.contains(
                    search_variable, case=False, regex=False
                ).any(),
                axis=1,
            )
        ]

    st.dataframe(
        variable_table,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.header("Tech Stack Used")

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.markdown(
            """
            - **Apache Spark:** Large-scale data processing
            - **PySpark:** Python interface for Spark
            - **Spark MLlib:** Machine learning models
            """
        )

    with tech_col2:
        st.markdown(
            """
            - **Python:** Project implementation
            - **Streamlit:** Interactive dashboard
            - **Pandas and Plotly:** Tables and visualisations
            """
        )

    st.divider()

    st.header("Machine Learning Models")

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:
        st.markdown("**Logistic Regression**")
        st.caption("Predicts the probability of a delay.")

    with model_col2:
        st.markdown("**Decision Tree**")
        st.caption("Makes predictions using feature-based decisions.")

    with model_col3:
        st.markdown("**Random Forest**")
        st.caption("Combines multiple decision trees.")

    st.caption(
        "Use the sidebar to explore model performance, delay insights, "
        "and the interactive prediction tool."
    )




# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":
    set_page_background("analytics_background.png")
    st.header("📊 Journey Delay Overview")
    st.caption(
        "A summary of railway journey punctuality and delay patterns."
    )
    

    if "is_delayed" not in df.columns:
        st.error("The dataset does not contain is_delayed.")
        st.stop()

    total = df.count()
    delayed = df.filter(F.col("is_delayed") == 1).count()
    on_time = df.filter(F.col("is_delayed") == 0).count()
    delay_rate = (delayed / total * 100) if total else 0
    on_time_rate = (on_time / total * 100) if total else 0

    # Key metrics
    st.subheader("Key Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Journeys", f"{total:,}")
    c2.metric("Delayed Journeys", f"{delayed:,}")
    c3.metric("On-Time Journeys", f"{on_time:,}")
    c4.metric("Delay Rate", f"{delay_rate:.1f}%")

    
    st.subheader("Training and Testing Split(80:20)")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Training Records (80%)",
            f"{1200102:,}"
        )

    with col2:
        st.metric(
            "Testing Records (20%)",
            f"{299898:,}"
        )
    

    st.divider()

    # Journey status charts
    st.subheader("Journey Punctuality")

    left, right = st.columns(2, gap="large")

    status_df = pd.DataFrame({
        "Journey Status": ["Delayed", "On time"],
        "Journeys": [delayed, on_time],
    })

    with left:
        fig = px.pie(
            status_df,
            names="Journey Status",
            values="Journeys",
            title="Distribution of Journey Status",
            hole=0.55,
            color="Journey Status",
            color_discrete_map={
                "Delayed": "#F97316",
                "On time": "#38BDF8",
            },
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate="%{label}<br>Journeys: %{value:,}<extra></extra>",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Status",
            margin=dict(l=20, r=20, t=65, b=20),
            height=380,
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.bar(
            status_df,
            x="Journey Status",
            y="Journeys",
            title="Journey Count by Status",
            text="Journeys",
            color="Journey Status",
            color_discrete_map={
                "Delayed": "#F97316",
                "On time": "#38BDF8",
            },
        )

        fig.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Journeys: %{y:,}<extra></extra>",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            yaxis_title="Number of Journeys",
            xaxis_title="",
            margin=dict(l=20, r=20, t=65, b=20),
            height=380,
        )

        st.plotly_chart(fig, use_container_width=True)

    # Dataset preview
    st.divider()
    st.subheader("🗂️ Dataset Preview")
    st.caption(
        "Explore a sample of the journey records used in this project."
    )

    preview_count = st.selectbox(
        "Number of rows to display",
        [5, 10, 15, 20],
        index=1,
        key="overview_preview_count",
    )

    st.dataframe(
        df.limit(preview_count).toPandas(),
        use_container_width=True,
        hide_index=True,
    )

    # Interpretation
    st.divider()
    st.subheader("Key Takeaway")

    st.markdown(
        f"""
        - **{delay_rate:.1f}%** of journeys in the dataset are labelled delayed.
        - **{on_time_rate:.1f}%** are on time or delayed by 15 minutes or less.
        - The prediction target is **arrival delay greater than 15 minutes**.
        """
    )

    st.caption(
        "Statistics are calculated from the loaded dataset. "
        "The charts show the distribution of the target label."
    )

# --------------------------------------------------
# BIG DATA TECHNOLOGIES
# --------------------------------------------------

elif page == "Big Data Technologies":
    set_page_background("tech.png")

    # Page styling: dark cards with readable white text
    st.markdown("""
    <style>
    .tech-card {
        background: rgba(12, 17, 29, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        margin-bottom: 12px;
    }

    .tech-card h3 {
        color: #FFFFFF;
        font-size: 20px;
        margin-top: 0;
        margin-bottom: 10px;
    }

    .tech-card p, .tech-card li {
        color: #E5E7EB;
        font-size: 15px;
        line-height: 1.65;
    }

    .tech-label {
        color: #67E8F9;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .workflow-step {
        background: rgba(12, 17, 29, 0.94);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
        min-height: 150px;
    }

    .workflow-number {
        color: #67E8F9;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .workflow-title {
        color: #FFFFFF;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .workflow-description {
        color: #E5E7EB;
        font-size: 13px;
        line-height: 1.5;
    }

    .workflow-arrow {
        color: #67E8F9;
        font-size: 26px;
        text-align: center;
        padding-top: 45px;
    }

    .section-note {
        color: #F3F4F6;
        font-size: 15px;
        line-height: 1.7;
    }

    div[data-testid="stExpander"] {
        background: rgba(12, 17, 29, 0.88);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # --------------------------------------------------
    # PAGE INTRODUCTION
    # --------------------------------------------------

    st.title("Big Data Technologies")
    st.markdown(
        """
        <p class="section-note">
        This page explains how Apache Spark, Spark SQL and PySpark MLlib
        support our Railway Delay Prediction project, from processing
        journey records to building machine-learning models.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # --------------------------------------------------
    # PROJECT WORKFLOW
    # --------------------------------------------------

    st.header("How the Project Works")
    st.caption(
        "The journey from raw railway records to a delay-risk prediction."
    )

    workflow = [
        (
            "01",
            "Railway Dataset",
            "Load journey records and inspect the available columns."
        ),
        (
            "02",
            "Apache Spark",
            "Process records using Spark DataFrames and transformations."
        ),
        (
            "03",
            "Spark SQL",
            "Query structured data and summarize journey patterns."
        ),
        (
            "04",
            "PySpark MLlib",
            "Prepare features and train classification models."
        ),
        (
            "05",
            "Prediction",
            "Use a trained model to classify a journey's delay risk."
        ),
    ]

    # Five connected stages, displayed in one row on wide screens
    workflow_columns = st.columns([1, 0.16, 1, 0.16, 1, 0.16, 1, 0.16, 1])

    positions = [0, 2, 4, 6, 8]

    for i, (number, title, description) in enumerate(workflow):
        with workflow_columns[positions[i]]:
            st.markdown(
                f"""
                <div class="workflow-step">
                    <div class="workflow-number">STEP {number}</div>
                    <div class="workflow-title">{title}</div>
                    <div class="workflow-description">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if i < len(workflow) - 1:
            with workflow_columns[positions[i] + 1]:
                st.markdown(
                    '<div class="workflow-arrow">→</div>',
                    unsafe_allow_html=True,
                )

    st.markdown(
        """
        <p class="section-note">
        <b>Final output:</b> The selected model predicts whether the
        arrival delay is likely to exceed 15 minutes. The dashboard
        then presents model results and journey insights.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # --------------------------------------------------
    # TECHNOLOGY 1: APACHE SPARK
    # --------------------------------------------------

    st.header("1. Apache Spark and PySpark")

    col1, col2 = st.columns([1.3, 1], gap="large")

    with col1:
        st.markdown("""
        <div class="tech-card">
            <div class="tech-label">Data Processing</div>
            <h3>Apache Spark</h3>
            <p>
            Apache Spark is a data-processing framework designed to
            handle large datasets efficiently. PySpark provides its
            Python interface, allowing us to work with Spark using Python.
            </p>
            <p><b>How it helps this project</b></p>
            <ul>
                <li>Loads the railway CSV into a Spark DataFrame.</li>
                <li>Provides operations to inspect and transform records.</li>
                <li>Supports filtering, grouping and calculating summaries.</li>
                <li>Provides the data-processing foundation for MLlib.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Example: Load the dataset")

        st.code(
            '''df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("ir_train.csv")
)

df.show(5)
df.printSchema()''',
            language="python",
        )

        st.caption(
            "This is an illustrative example. The deployed application "
            "may obtain the CSV from a remote dataset source."
        )

    st.divider()

    # --------------------------------------------------
    # TECHNOLOGY 2: SPARK SQL
    # --------------------------------------------------

    st.header("2. Spark SQL")

    col1, col2 = st.columns([1.3, 1], gap="large")

    with col1:
        st.markdown("""
        <div class="tech-card">
            <div class="tech-label">Structured Data Analysis</div>
            <h3>Spark SQL</h3>
            <p>
            Spark SQL lets us query structured data using familiar SQL
            statements. It works with Spark DataFrames and can summarize
            large collections of records.
            </p>
            <p><b>How it helps this project</b></p>
            <ul>
                <li>Counts delayed and non-delayed journeys.</li>
                <li>Groups records by categories for analysis.</li>
                <li>Helps identify patterns in the dataset.</li>
                <li>Makes data analysis easier to demonstrate and explain.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Example: Count journey status")

        st.code(
            '''df.createOrReplaceTempView("railway_data")

delay_counts = spark.sql("""
    SELECT is_delayed, COUNT(*) AS total
    FROM railway_data
    GROUP BY is_delayed
    ORDER BY is_delayed
""")

delay_counts.show()''',
            language="python",
        )

        st.caption(
            "`is_delayed = 0` means the delay does not exceed 15 minutes; "
            "`is_delayed = 1` means it exceeds 15 minutes."
        )

    st.divider()

    # --------------------------------------------------
    # TECHNOLOGY 3: PYSPARK MLLIB
    # --------------------------------------------------

    st.header("3. PySpark MLlib")

    st.markdown("""
    <p class="section-note">
    PySpark MLlib provides machine-learning algorithms and tools that
    work with Spark DataFrames. In this project, it supports the
    classification task of predicting whether a journey's arrival
    delay will exceed 15 minutes.
    </p>
    """, unsafe_allow_html=True)

    model_cols = st.columns(3, gap="medium")

    model_info = [
        (
            "Logistic Regression",
            "Estimates the probability of a journey belonging to the delayed class."
        ),
        (
            "Decision Tree",
            "Uses feature-based decision rules to classify a journey."
        ),
        (
            "Random Forest",
            "Combines multiple decision trees to make a classification."
        ),
    ]

    for col, (name, description) in zip(model_cols, model_info):
        with col:
            st.markdown(
                f"""
                <div class="tech-card">
                    <div class="tech-label">MLlib Model</div>
                    <h3>{name}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Example: Build an MLlib pipeline")

    with st.expander("View illustrative training code"):
        st.code(
            '''from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

classifier = LogisticRegression(
    featuresCol="features",
    labelCol="is_delayed"
)

pipeline = Pipeline(
    stages=[assembler, classifier]
)

model = pipeline.fit(train_df)''',
            language="python",
        )

        st.caption(
            "This is a simplified example, not a replacement for the "
            "actual training pipeline. The project's saved preprocessing "
            "pipeline and trained models are loaded by the dashboard."
        )

    st.markdown("""
    <div class="tech-card">
        <div class="tech-label">Why MLlib?</div>
        <h3>From Data to Prediction</h3>
        <p>
        MLlib provides the classification algorithms and pipeline
        components. After training, the saved preprocessing pipeline
        and model can be loaded to process new journey details and
        generate a prediction without retraining the model every time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --------------------------------------------------
    # TECHNOLOGY SUMMARY
    # --------------------------------------------------

    st.header("Role of Each Technology")

    summary_df = pd.DataFrame({
        "Technology": [
            "Apache Spark / PySpark",
            "Spark SQL",
            "PySpark MLlib",
            "Streamlit",
        ],
        "Main Role": [
            "Load and process railway records",
            "Query and summarize structured data",
            "Train and run delay classification models",
            "Display analysis and predictions through a dashboard",
        ],
    })

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("""
    <div class="tech-card">
        <h3>Key Takeaway</h3>
        <p>
        Spark provides the data-processing foundation, Spark SQL supports
        structured analysis, and MLlib provides the machine-learning
        models. Together with Streamlit, these technologies help turn
        railway journey records into an interactive delay-prediction system.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":
    set_page_background("model_comparision.png")
    st.header("📈 Model Performance Comparison")
    st.caption(
        "Compare Logistic Regression, Decision Tree, and Random Forest "
        "using the evaluation metrics."
    )

    if not isinstance(metrics, dict) or not metrics:
        st.warning("No evaluation metrics were found in metrics.json.")
        st.stop()

    rows = []

    for model_name, values in metrics.items():
        if isinstance(values, dict):
            row = {"Model": model_name.replace("_", " ").title()}
            row.update(values)
            rows.append(row)

    if not rows:
        st.warning(
            "The metrics file has an unexpected structure. "
            "Check models/metrics.json."
        )
        st.stop()

    metric_df = pd.DataFrame(rows)

    numeric_metrics = [
        c for c in metric_df.columns
        if c != "Model"
        and pd.api.types.is_numeric_dtype(metric_df[c])
    ]

    if not numeric_metrics:
        st.warning("No numeric evaluation metrics were found.")
        st.stop()

    # Model selection
    st.subheader("Select Models to View")

    model_options = ["All Models"] + metric_df["Model"].tolist()

    selected_model = st.selectbox(
        "Choose a model",
        model_options,
        key="comparison_model_select",
    )

    if selected_model == "All Models":
        display_df = metric_df.copy()
    else:
        display_df = metric_df[
            metric_df["Model"] == selected_model
        ].copy()

    # Metric cards
    st.divider()
    st.subheader("Performance Metrics")

    preferred_metrics = [
        "accuracy",
        "weightedPrecision",
        "weightedRecall",
        "weightedFMeasure",
        "areaUnderROC",
    ]

    metric_mapping = {
        "accuracy": "Accuracy",
        "weightedPrecision": "Weighted Precision",
        "weightedRecall": "Weighted Recall",
        "weightedFMeasure": "Weighted F1-Score",
        "areaUnderROC": "ROC-AUC",
    }

    available_metrics = [
        m for m in preferred_metrics if m in numeric_metrics
    ]

    if not available_metrics:
        available_metrics = numeric_metrics[:5]

    metric_cols = st.columns(len(available_metrics))

    for i, metric in enumerate(available_metrics):
        best_idx = metric_df[metric].idxmax()
        best_model = metric_df.loc[best_idx, "Model"]
        value = display_df[metric].iloc[0] if len(display_df) == 1 else None

        with metric_cols[i]:
            if value is not None:
                st.metric(
                    metric_mapping.get(metric, metric),
                    f"{value:.3f}",
                )
            else:
                st.metric(
                    metric_mapping.get(metric, metric),
                    f"{metric_df[metric].max():.3f}",
                    help=f"Best model: {best_model}",
                )

    # Full comparison table
    st.divider()
    st.subheader("Detailed Results")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # Chart comparison
    st.divider()
    st.subheader("Visual Comparison")

    selected_metric = st.selectbox(
        "Choose a metric to compare",
        numeric_metrics,
        format_func=lambda x: metric_mapping.get(x, x),
        key="comparison_metric_select",
    )

    fig = px.bar(
        display_df,
        x="Model",
        y=selected_metric,
        color="Model",
        text=selected_metric,
        title=f"{metric_mapping.get(selected_metric, selected_metric)} by Model",
        color_discrete_sequence=["#38BDF8", "#F97316", "#A78BFA"],
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis_title=metric_mapping.get(selected_metric, selected_metric),
        xaxis_title="",
        yaxis_range=[
            0,
            min(1.1, max(1.0, display_df[selected_metric].max() * 1.12))
        ],
        margin=dict(l=20, r=20, t=65, b=20),
        height=420,
    )

    chart_col1, chart_col2, chart_col3 = st.columns([2, 5, 2])

    with chart_col2:
        st.plotly_chart(fig, use_container_width=True)

    # Best model for the selected metric
    best_idx = metric_df[selected_metric].idxmax()
    best_model = metric_df.loc[best_idx, "Model"]
    best_score = metric_df.loc[best_idx, selected_metric]

    st.info(
        f"**Best model for {metric_mapping.get(selected_metric, selected_metric)}:** "
        f"{best_model} ({best_score:.3f})"
    )

    st.caption(
        "Metrics are loaded from metrics.json and are not recalculated "
        "when the dashboard opens. Higher values indicate better scores "
        "for the metrics displayed here."
    )

    

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":
    set_page_background("model_evaluation.png")
    st.header("🧪 Model Evaluation")
    st.caption(
        "Understand how accurately each model predicts railway delays "
        "and where its predictions go wrong."
    )

    evaluation_path = MODEL_DIR / "evaluation_results.json"

    if not evaluation_path.exists():
        st.error(
            "evaluation_results.json was not found in the models folder."
        )
        st.stop()

    with open(evaluation_path, encoding="utf-8") as f:
        evaluation_results = json.load(f)

    if not evaluation_results:
        st.warning("No evaluation results are available.")
        st.stop()

    # Model selector
    st.subheader("Select a Model")

    model_names = list(evaluation_results.keys())

    selected_model = st.selectbox(
        "Choose a model to evaluate",
        model_names,
        format_func=lambda x: x.replace("_", " ").title(),
        key="evaluation_model",
    )

    model_results = evaluation_results[selected_model]
    cm = model_results["confusion_matrix"]
    report = model_results["classification_report"]

    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    # Key metrics
    st.divider()
    st.subheader("Performance Summary")

    accuracy = report.get("accuracy", 0)

    delayed_recall = (
        report.get("Delayed", {}).get("recall", 0)
    )
    delayed_f1 = (
        report.get("Delayed", {}).get("f1-score", 0)
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Overall Accuracy", f"{accuracy:.2%}")
    c2.metric("Correct Predictions", f"{tn + tp:,}")
    c3.metric("Delay Detection Rate", f"{delayed_recall:.2%}")
    c4.metric("Delayed F1-Score", f"{delayed_f1:.3f}")

    # Confusion matrix
    st.divider()
    st.subheader("Confusion Matrix")
    st.caption(
        "Rows represent the actual status. Columns represent the "
        "status predicted by the model."
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual: On time", "Actual: Delayed"],
        columns=["Predicted: On time", "Predicted: Delayed"],
    )

    fig = px.imshow(
        cm_df,
        text_auto=",",
        aspect="auto",
        color_continuous_scale=[
            [0, "#172554"],
            [0.5, "#2563EB"],
            [1, "#7DD3FC"],
        ],
        labels={
            "x": "Predicted Status",
            "y": "Actual Status",
            "color": "Journeys",
        },
        title=f"{selected_model.replace('_', ' ').title()} — Confusion Matrix",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=70, b=30),
        height=350,
        font=dict(size=12),
    )

    fig.update_xaxes(side="bottom")

    cm_left, cm_center, cm_right = st.columns([1, 5, 1])

    with cm_center:
        st.plotly_chart(fig, use_container_width=True)

    # Prediction counts
    st.divider()
    st.subheader("Prediction Breakdown")

    p1, p2, p3, p4 = st.columns(4)

    p1.metric("Correctly Identified On-Time", f"{tn:,}")
    p2.metric("False Delay Warnings", f"{fp:,}")
    p3.metric("Missed Delays", f"{fn:,}")
    p4.metric("Correctly Detected Delays", f"{tp:,}")

    # Classification report
    st.divider()
    st.subheader("Classification Report")
    st.caption(
        "Precision measures prediction correctness, recall measures "
        "how many actual cases were identified, and F1-score balances both."
    )

    report_rows = []

    for label in ["On time", "Delayed", "macro avg", "weighted avg"]:
        if label in report:
            values = report[label]

            report_rows.append({
                "Class": label.title(),
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1-Score": values["f1-score"],
                "Support": int(values["support"]),
            })

    report_df = pd.DataFrame(report_rows)

    st.dataframe(
        report_df.style.format({
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1-Score": "{:.3f}",
            "Support": "{:,}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Error analysis
    st.divider()
    st.subheader("Error Analysis")

    error_left, error_right = st.columns(2, gap="large")

    with error_left:
        st.markdown("#### False Positives")
        st.metric("False Delay Warnings", f"{fp:,}")
        st.write(
            "These journeys were actually on time but the model "
            "predicted that they would be delayed."
        )

    with error_right:
        st.markdown("#### False Negatives")
        st.metric("Missed Delays", f"{fn:,}")
        st.write(
            "These journeys were actually delayed but the model "
            "predicted that they would be on time."
        )

    st.info(
        "Missed delays can prevent passengers and railway staff from "
        "receiving an early warning. False delay warnings can lead to "
        "unnecessary precautions."
    )

    st.caption(
        "Results are taken from the saved Colab evaluation. "
        "The dashboard does not retrain or reevaluate the models."
    )




# --------------------------------------------------
# DELAY INSIGHTS
# --------------------------------------------------

elif page == "Delay Insights":
    set_page_background("delay_concept.png")
    st.header("🔍 Railway Delay Insights")
    st.caption(
        "Explore how delay rates vary by railway zone, train type, "
        "departure hour, and month."
    )

    if "is_delayed" not in df.columns:
        st.error("The dataset does not contain is_delayed.")
        st.stop()

    possible_columns = [
        c for c in [
            "zone",
            "railway_zone",
            "train_type",
            "hour",
            "departure_hour",
            "month",
            "departure_month",
        ]
        if c in df.columns
    ]

    if not possible_columns:
        st.warning(
            "No recognized zone, train type, hour, or month column "
            "was found."
        )
        st.write(df.columns)
        st.stop()

    # Filters
    st.subheader("Explore Delay Patterns")

    filter1, filter2 = st.columns(2)

    with filter1:
        selected_column = st.selectbox(
            "Group journeys by",
            possible_columns,
            format_func=lambda x: x.replace("_", " ").title(),
            key="insights_group_column",
        )

    with filter2:
        sort_option = st.selectbox(
            "Sort results by",
            [
                "Highest delay rate first",
                "Lowest delay rate first",
                "Most journeys first",
                "Fewest journeys first",
                "Group name (A–Z)",
            ],
            key="insights_sort_option",
        )

    filter3, filter4 = st.columns(2)

    with filter3:
        min_journeys = st.number_input(
            "Minimum journeys in a group",
            min_value=1,
            max_value=100000,
            value=100,
            step=100,
            help="Groups with fewer journeys are excluded.",
        )

    with filter4:
        min_delay_rate = st.slider(
            "Minimum delay rate (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
        )

    # Aggregate data
    summary = (
        df.groupBy(selected_column)
        .agg(
            F.count("*").alias("Total Journeys"),
            F.sum(
                F.when(F.col("is_delayed") == 1, 1).otherwise(0)
            ).alias("Delayed Journeys"),
        )
        .withColumn(
            "Delay Rate (%)",
            F.round(
                F.col("Delayed Journeys")
                / F.col("Total Journeys") * 100,
                2,
            ),
        )
        .filter(F.col(selected_column).isNotNull())
        .filter(F.col("Total Journeys") >= min_journeys)
        .filter(F.col("Delay Rate (%)") >= min_delay_rate)
    )

    # Sorting
    if sort_option == "Highest delay rate first":
        summary = summary.orderBy(F.desc("Delay Rate (%)"))
    elif sort_option == "Lowest delay rate first":
        summary = summary.orderBy(F.asc("Delay Rate (%)"))
    elif sort_option == "Most journeys first":
        summary = summary.orderBy(F.desc("Total Journeys"))
    elif sort_option == "Fewest journeys first":
        summary = summary.orderBy(F.asc("Total Journeys"))
    else:
        summary = summary.orderBy(F.asc(selected_column))

    summary_pd = summary.limit(50).toPandas()

    if summary_pd.empty:
        st.warning(
            "No groups match these filters. Try lowering the minimum "
            "journey count or minimum delay rate."
        )
        st.stop()

    # Summary metrics
    st.divider()
    st.subheader("Summary")

    total_groups = len(summary_pd)
    highest_delay = summary_pd.loc[
        summary_pd["Delay Rate (%)"].idxmax()
    ]
    lowest_delay = summary_pd.loc[
        summary_pd["Delay Rate (%)"].idxmin()
    ]

    m1, m2, m3 = st.columns(3)

    m1.metric("Groups Displayed", f"{total_groups:,}")
    m2.metric(
        "Highest Delay Rate",
        f"{highest_delay['Delay Rate (%)']:.2f}%",
        help=str(highest_delay[selected_column]),
    )
    m3.metric(
        "Lowest Delay Rate",
        f"{lowest_delay['Delay Rate (%)']:.2f}%",
        help=str(lowest_delay[selected_column]),
    )

    # Chart
    st.divider()
    st.subheader("Delay Rate Comparison")

    chart_type = st.radio(
        "Chart type",
        ["Bar chart", "Line chart"],
        horizontal=True,
        key="insights_chart_type",
    )

    chart_df = summary_pd.copy()
    chart_df[selected_column] = chart_df[selected_column].astype(str)

    if chart_type == "Bar chart":
        fig = px.bar(
            chart_df,
            x=selected_column,
            y="Delay Rate (%)",
            color="Delay Rate (%)",
            color_continuous_scale=[
                "#2A9D8F",
                "#E9C46A",
                "#F4A261",
                "#E76F51",
            ],
            hover_data=["Total Journeys", "Delayed Journeys"],
            title=f"Delay Rate by {selected_column.replace('_', ' ').title()}",
            text="Delay Rate (%)",
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            cliponaxis=False,
        )
        fig.update_layout(coloraxis_colorbar_title="Delay Rate (%)")

    else:
        fig = px.line(
            chart_df,
            x=selected_column,
            y="Delay Rate (%)",
            markers=True,
            color_discrete_sequence=["#E9A23B"],
            hover_data=["Total Journeys", "Delayed Journeys"],
            title=f"Delay Rate by {selected_column.replace('_', ' ').title()}",
        )

        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=8),
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=selected_column.replace("_", " ").title(),
        yaxis_title="Delay Rate (%)",
        yaxis_range=[0, 105],
        margin=dict(l=20, r=25, t=65, b=30),
        height=430,
    )

    # Keep the chart from stretching across the entire page.
    chart_left, chart_center, chart_right = st.columns([1, 6, 1])

    with chart_center:
        st.plotly_chart(fig, use_container_width=True)

    # Table
    st.divider()
    st.subheader("Detailed Delay Statistics")

    st.dataframe(
        summary_pd,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Delay Rate (%)": st.column_config.NumberColumn(
                "Delay Rate (%)",
                format="%.2f%%",
            ),
            "Total Journeys": st.column_config.NumberColumn(
                "Total Journeys",
                format="%d",
            ),
            "Delayed Journeys": st.column_config.NumberColumn(
                "Delayed Journeys",
                format="%d",
            ),
        },
    )

    st.caption(
        "The chart and table show up to 50 groups after filtering. "
        "Delay rate is the percentage of journeys labelled as delayed."
    )


# --------------------------------------------------
# DELAY  PREDICTOR
# --------------------------------------------------

elif page == "Train Delay Predictor":
    set_page_background("prediction_workflow.png")
    st.header("🚆 Train Delay Predictor")
    st.write(
        "Enter the planned journey details to estimate "
        "whether the train is likely to be delayed."
    )

    selected_model = st.selectbox(
        "Select Model",
        list(models.keys()),
    )

    categorical_columns = metadata.get("categorical_columns", [])
    numeric_columns = metadata.get("numeric_columns", [])

    with st.form("prediction_form"):
        st.subheader("Train and Route Details")

        c1, c2, c3 = st.columns(3)

        with c1:
            train_type = st.selectbox(
                "Train Type",
                ["Express", "Superfast", "Passenger", "Mail", "Other"],
            )
            train_number = st.number_input(
                "Train Number", min_value=1, value=12001
            )
            zone = st.selectbox(
                "Railway Zone",
                ["Northern Railway (NR)", "Southern Railway (SR)",
                 "Central Railway (CR)", "Western Railway (WR)",
                 "Eastern Railway (ER)", "South Central Railway (SCR)",
                 "South Eastern Railway (SER)",
                 "North Central Railway (NCR)",
                 "North Eastern Railway (NER)",
                 "North Western Railway (NWR)",
                 "West Central Railway (WCR)",
                 "East Central Railway (ECR)",
                 "East Coast Railway (ECoR)",
                 "South Western Railway (SWR)",
                 "Northeast Frontier Railway (NFR)"],
            )
            zone_abbr = st.text_input("Zone Abbreviation", value="NR")

        with c2:
            season = st.selectbox(
                "Season", ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
            )
            departure_hour = st.slider(
                "Departure Hour", 0, 23, 10
            )
            day_of_week = st.selectbox(
                "Day of Week (0 = Monday)", list(range(7))
            )
            distance_km = st.number_input(
                "Distance (km)", min_value=1.0, value=500.0
            )

        with c3:
            source_station_category = st.selectbox(
                "Source Station Category",
                ["A1", "A", "B", "C", "D", "Other"],
            )
            destination_station_category = st.selectbox(
                "Destination Station Category",
                ["A1", "A", "B", "C", "D", "Other"],
            )
            traction_type = st.selectbox(
                "Traction Type", ["Electric", "Diesel", "Other"]
            )
            num_scheduled_stops = st.number_input(
                "Scheduled Stops", min_value=0, value=8
            )

        st.subheader("Schedule and Operating Conditions")

        c1, c2, c3 = st.columns(3)

        with c1:
            year = st.number_input(
                "Journey Year", min_value=2020, max_value=2100, value=2026
            )
            month = st.slider("Month", 1, 12, 10)
            scheduled_travel_hours = st.number_input(
                "Scheduled Travel Hours", min_value=0.1, value=8.0
            )
            is_weekend = st.selectbox("Weekend?", [0, 1])

        with c2:
            is_night_departure = st.selectbox("Night Departure?", [0, 1])
            is_peak_hour = st.selectbox("Peak Hour?", [0, 1])
            is_festival_season = st.selectbox("Festival Season?", [0, 1])
            track_doubled = st.selectbox("Track Doubled?", [0, 1])

        with c3:
            is_hdn_route = st.selectbox("High-Density Network Route?", [0, 1])
            is_electrified = st.selectbox("Electrified Route?", [0, 1])
            psr_count = st.number_input(
                "Permanent Speed Restrictions", min_value=0, value=0
            )
            is_circular_route = st.selectbox("Circular Route?", [0, 1])

        st.subheader("Weather, Rolling Stock and Historical Conditions")

        c1, c2, c3 = st.columns(3)

        with c1:
            is_monsoon_season = st.selectbox("Monsoon Conditions?", [0, 1])
            is_fog_risk = st.selectbox("Fog Risk?", [0, 1])
            fog_risk_score = st.slider("Fog Risk Score", 0.0, 1.0, 0.2)
            zone_fog_index = st.slider("Zone Fog Index", 0.0, 1.0, 0.2)
            zone_congestion_index = st.slider(
                "Zone Congestion Index", 0.0, 1.0, 0.5
            )
            season_severity_score = st.slider(
                "Season Severity Score", 0.0, 1.0, 0.5
            )

        with c2:
            loco_age_years = st.number_input(
                "Locomotive Age (years)", min_value=0.0, value=8.0
            )
            coach_age_years = st.number_input(
                "Coach Age (years)", min_value=0.0, value=8.0
            )
            has_lhb_coaches = st.selectbox("LHB Coaches?", [0, 1])
            is_rake_shared = st.selectbox("Shared Rake?", [0, 1])
            maintenance_score = st.slider(
                "Maintenance Score", 0.0, 1.0, 0.8
            )

        with c3:
            seat_utilisation_pct = st.slider(
                "Seat Utilisation (%)", 0.0, 100.0, 75.0
            )
            is_overloaded = st.selectbox("Overloaded?", [0, 1])
            late_incoming_rake = st.selectbox(
                "Incoming Rake Already Late?", [0, 1]
            )
            is_special_train = st.selectbox("Special Train?", [0, 1])
            route_historical_ontime_pct = st.slider(
                "Historical On-Time Rate (%)", 0.0, 100.0, 75.0
            )

        submitted = st.form_submit_button("Predict Delay Risk")

    
    if submitted:
        import pandas as pd
        import time
        import random
        from pyspark.sql import SparkSession

        # Build feature row using the training column names.
        input_values = {
            "train_type": train_type,
            "season": season,
            "zone": zone,
            "zone_abbr": zone_abbr,
            "source_station_category": source_station_category,
            "destination_station_category": destination_station_category,
            "traction_type": traction_type,
            "train_number": train_number,
            "year": year,
            "month": month,
            "day_of_week": day_of_week,
            "departure_hour": departure_hour,
            "is_weekend": is_weekend,
            "is_night_departure": is_night_departure,
            "is_peak_hour": is_peak_hour,
            "is_festival_season": is_festival_season,
            "distance_km": distance_km,
            "num_scheduled_stops": num_scheduled_stops,
            "scheduled_travel_hours": scheduled_travel_hours,
            "track_doubled": track_doubled,
            "is_hdn_route": is_hdn_route,
            "is_electrified": is_electrified,
            "psr_count": psr_count,
            "is_circular_route": is_circular_route,
            "is_monsoon_season": is_monsoon_season,
            "is_fog_risk": is_fog_risk,
            "fog_risk_score": fog_risk_score,
            "zone_fog_index": zone_fog_index,
            "zone_congestion_index": zone_congestion_index,
            "season_severity_score": season_severity_score,
            "loco_age_years": loco_age_years,
            "coach_age_years": coach_age_years,
            "has_lhb_coaches": has_lhb_coaches,
            "is_rake_shared": is_rake_shared,
            "maintenance_score": maintenance_score,
            "seat_utilisation_pct": seat_utilisation_pct,
            "is_overloaded": is_overloaded,
            "late_incoming_rake": late_incoming_rake,
            "is_special_train": is_special_train,
            "route_historical_ontime_pct": route_historical_ontime_pct,
        }

        required_columns = categorical_columns + numeric_columns
        missing = [col for col in required_columns if col not in input_values]

        if missing:
            st.error(f"Missing required input fields: {missing}")

        else:
            input_df = pd.DataFrame([
                {col: input_values[col] for col in required_columns}
            ])

            # Game-style prediction progress display.
            st.subheader("Analyzing Journey")

            progress_bar = st.progress(0)
            status_text = st.empty()

            stages = [
                (15, "Initializing railway analytics system..."),
                (30, "Loading journey features..."),
                (45, "Analyzing route and operating conditions..."),
                (60, "Running the preprocessing pipeline..."),
                (75, f"Running {selected_model} model..."),
                (90, "Preparing prediction and risk assessment..."),
            ]

            try:
                status_text.write(stages[0][1])
                progress_bar.progress(5)

                

                spark_input = spark.createDataFrame(input_df)

                status_text.write(stages[3][1])
                progress_bar.progress(60)

                transformed = preprocessor.transform(spark_input)

                status_text.write(stages[4][1])
                progress_bar.progress(75)

                model = models[selected_model]
                result = model.transform(transformed).select(
                    "prediction"
                ).first()

                prediction = int(result["prediction"])

                status_text.write(stages[5][1])
                progress_bar.progress(100)
                status_text.write("Analysis complete.")

                # Prediction result.
                st.divider()
                st.subheader("Prediction Result")

                if prediction == 1:
                    st.error("Predicted Status: DELAYED")
                    st.write(
                        "The model predicts that arrival delay may exceed "
                        "15 minutes."
                    )
                else:
                    st.success("Predicted Status: ON TIME")
                    st.write(
                        "The model predicts that arrival delay will not exceed "
                        "15 minutes."
                    )

                # Possible contributing factors based on input values.
                # These are rules, not explanations from the ML model.
                possible_reasons = []

                if late_incoming_rake == 1:
                    possible_reasons.append(
                        "Late incoming rake: the train may start its journey late "
                        "because the incoming rake is already delayed."
                    )

                if is_fog_risk == 1 or fog_risk_score >= 0.7:
                    possible_reasons.append(
                        "Fog risk: poor visibility may require slower train "
                        "operations."
                    )

                if is_monsoon_season == 1:
                    possible_reasons.append(
                        "Monsoon conditions: heavy rain or waterlogging may "
                        "affect railway operations."
                    )

                if zone_congestion_index >= 0.7:
                    possible_reasons.append(
                        "High zone congestion: busy routes may experience "
                        "additional waiting or traffic."
                    )

                if is_peak_hour == 1:
                    possible_reasons.append(
                        "Peak-hour departure: heavier traffic may increase "
                        "the chance of operational delays."
                    )

                if psr_count >= 3:
                    possible_reasons.append(
                        "Multiple speed restrictions: permanent speed restrictions "
                        "may increase travel time."
                    )

                if num_scheduled_stops >= 15:
                    possible_reasons.append(
                        "Many scheduled stops: additional stops can increase "
                        "the journey's exposure to timing disruptions."
                    )

                if distance_km >= 800:
                    possible_reasons.append(
                        "Long-distance journey: a longer route has more "
                        "opportunities for disruptions to affect arrival time."
                    )

                if maintenance_score < 0.5:
                    possible_reasons.append(
                        "Low maintenance score: the supplied maintenance indicator "
                        "suggests a potential operational concern."
                    )

                if season_severity_score >= 0.7:
                    possible_reasons.append(
                        "Severe seasonal conditions: the supplied season severity "
                        "score indicates potentially challenging conditions."
                    )

                if zone_fog_index >= 0.7:
                    possible_reasons.append(
                        "High zone fog index: fog-related disruption may be more "
                        "likely in the selected zone."
                    )

                if route_historical_ontime_pct < 60:
                    possible_reasons.append(
                        "Low historical on-time rate: this route has a relatively "
                        "low supplied on-time performance indicator."
                    )

                if is_overloaded == 1:
                    possible_reasons.append(
                        "Overloaded train: the supplied overload indicator may "
                        "represent an operating constraint."
                    )

                if possible_reasons:
                    st.subheader("Possible Delay Contributing Factors")

                    for reason in possible_reasons:
                        st.markdown(f"- {reason}")

                else:
                    st.subheader("Possible Delay Contributing Factors")
                    st.write(
                        "No obvious contributing factors were identified by "
                        "the dashboard's rule-based checks. A predicted delay "
                        "can still occur due to factors not included here."
                    )

                st.caption(
                    "The prediction is based on the selected machine-learning "
                    "model. The contributing factors are rule-based suggestions, "
                    "not confirmed causes or model-generated explanations."
                )

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Prediction failed: {e}")
                st.info(
                    "Check that the saved preprocessing pipeline, input columns, "
                    "and selected model are compatible."
                )


# --------------------------------------------------
# BOTTOM PAGE NAVIGATION
# --------------------------------------------------

st.divider()

current_index = pages.index(page)

prev_col, info_col, next_col = st.columns([1, 2, 1])

with prev_col:
    if current_index > 0:
        st.button(
            "← Previous",
            key="bottom_previous",
            use_container_width=True,
            on_click=navigate_to_page,
            args=(pages[current_index - 1],),
        )

with info_col:
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding-top: 8px;
            color: #FFFFFF;
            font-size: 14px;
        ">
            Page {current_index + 1} of {len(pages)}
        </div>
        """,
        unsafe_allow_html=True,
    )

with next_col:
    if current_index < len(pages) - 1:
        st.button(
            "Next →",
            key="bottom_next",
            use_container_width=True,
            on_click=navigate_to_page,
            args=(pages[current_index + 1],),
        )

if page_changed:
    scroll_to_top()




# --------------------------------------------------
# FIX BOTTOM PREVIOUS / NEXT BUTTON STYLING
# --------------------------------------------------

st.markdown("""
<style>

/* Previous and Next buttons */
.st-key-bottom_previous button,
.st-key-bottom_next button {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    border: 1px solid #64748B !important;
    border-radius: 8px !important;
    min-height: 42px !important;
}

/* Keep button text visible */
.st-key-bottom_previous button *,
.st-key-bottom_next button * {
    color: #FFFFFF !important;
}

/* Disabled buttons */
.st-key-bottom_previous button:disabled,
.st-key-bottom_next button:disabled {
    background-color: #334155 !important;
    color: #CBD5E1 !important;
    border: 1px solid #475569 !important;
    opacity: 1 !important;
    cursor: not-allowed !important;
}

/* Hover effect for enabled buttons */
.st-key-bottom_previous button:not(:disabled):hover,
.st-key-bottom_next button:not(:disabled):hover {
    background-color: #2563EB !important;
    border-color: #60A5FA !important;
}

</style>
""", unsafe_allow_html=True)

