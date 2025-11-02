import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Food Distribution Analysis Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    h2 {
        color: #2c3e50;
        padding-top: 10px;
    }
    .highlight-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and introduction
st.title("🍽️ Food Distribution Analysis & Prediction System")
st.markdown("### Comprehensive ML-Powered Dashboard for Distribution Optimization")

# Hero metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📊 Total Records",
        value="58,016",
        delta="Ready for Analysis"
    )

with col2:
    st.metric(
        label="🤖 ML Models",
        value="29",
        delta="Trained & Evaluated"
    )

with col3:
    st.metric(
        label="🎯 Best Accuracy",
        value="99.95%",
        delta="Classification"
    )

with col4:
    st.metric(
        label="💰 Projected ROI",
        value="$1.5M",
        delta="3-Year Net Benefit"
    )

# Main content
st.markdown("---")

# Project overview
st.markdown("""
<div class="highlight-box">
    <h2>🎯 Project Overview</h2>
    <p>This dashboard presents a comprehensive analysis of food distribution data spanning <strong>7 years (2017-2023)</strong> 
    with <strong>284+ million units</strong> distributed across multiple regions. Using advanced machine learning techniques, 
    we've developed predictive models to optimize distribution efficiency and provide actionable business insights.</p>
</div>
""", unsafe_allow_html=True)

# Two-column layout for key highlights
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📈 **Analysis Phases Completed**")
    st.markdown("""
    - ✅ **Phase 1:** Data Understanding & Preparation
    - ✅ **Phase 2:** Exploratory Data Analysis (8 sections)
    - ✅ **Phase 3:** Feature Engineering (40+ features)
    - ✅ **Phase 4:** Machine Learning Modeling (29 models)
    - ✅ **Phase 5:** Model Interpretation & Insights
    """)
    
    st.markdown("#### 🎓 **Key Achievements**")
    st.markdown("""
    - 🏆 99.95% classification accuracy
    - 📊 83.58% R² score for regression
    - 🔍 SHAP-based model explainability
    - 💼 $300K-$500K annual savings potential
    """)

with col2:
    st.markdown("#### 🤖 **Model Portfolio**")
    st.markdown("""
    **Regression Models (12):** Linear Regression, Ridge, Lasso, ElasticNet, 
    Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, 
    CatBoost, SVR, KNN
    
    **Classification Models (10):** Logistic Regression, Decision Tree, 
    Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVC, 
    KNN, Naive Bayes
    
    **Clustering Models (4):** K-Means (k=3), K-Means (k=5), 
    Hierarchical, Gaussian Mixture
    
    **Time Series Models (3):** Naive, Moving Average, Exponential Smoothing
    """)

# Navigation guide
st.markdown("---")
st.markdown("### 🧭 Navigation Guide")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.info("**📊 Data Overview**\n\nExplore dataset statistics, data quality metrics, and sample distributions.")
    st.info("**📈 Exploratory Analysis**\n\nInteractive visualizations of temporal trends, correlations, and patterns.")

with nav_col2:
    st.success("**🤖 Model Performance**\n\nCompare all 29 models, view metrics, and analyze predictions.")
    st.success("**🔍 Model Insights**\n\nFeature importance, SHAP analysis, and error diagnostics.")

with nav_col3:
    st.warning("**💼 Business Impact**\n\nROI projections, efficiency analysis, and cost-benefit scenarios.")
    st.warning("**🎯 Recommendations**\n\nPrioritized action items with implementation timeline.")

# Quick stats summary
st.markdown("---")
st.markdown("### 📊 Quick Statistics")

stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)

with stat_col1:
    st.metric("Time Period", "7 Years", "2017-2023")

with stat_col2:
    st.metric("Total Distribution", "284M Units", "+16.2% Growth")

with stat_col3:
    st.metric("Features Engineered", "40+", "For Modeling")

with stat_col4:
    st.metric("Best R² Score", "0.8358", "Decision Tree")

with stat_col5:
    st.metric("Payback Period", "6-12 Months", "ML Investment")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🎓 <strong>Data Science Project made by Akash Shah</strong> | Built with Streamlit & Python</p>
    <p>📧 Analysis includes EDA, Feature Engineering, ML Modeling, and Business Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/food-bar.png", width=80)
    st.markdown("## 📊 Dashboard Navigation")
    st.markdown("Use the pages menu above to explore different sections.")
    
    st.markdown("---")
    st.markdown("### 🎯 Key Metrics")
    st.info("**Models Trained:** 29")
    st.info("**Best Accuracy:** 99.95%")
    st.info("**Data Records:** 58,016")
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This dashboard presents a comprehensive food distribution analysis 
    using machine learning and data science techniques.
    """)
