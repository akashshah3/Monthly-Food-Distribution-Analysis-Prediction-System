import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import load_raw_data, get_data_summary, get_column_info

st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")

st.title("📊 Data Overview & Quality Assessment")
st.markdown("### Comprehensive analysis of dataset structure and quality")

# Load data
df = load_raw_data()

if df is not None:
    # Summary statistics
    summary = get_data_summary(df)
    
    st.markdown("---")
    st.markdown("## 📈 Dataset Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", f"{summary['total_records']:,}")
    with col2:
        st.metric("Total Columns", summary['total_columns'])
    with col3:
        st.metric("Numeric Columns", summary['numeric_columns'])
    with col4:
        st.metric("Categorical Columns", summary['categorical_columns'])
    with col5:
        st.metric("Missing Values", f"{summary['missing_values']:,}")
    
    # Data quality metrics
    st.markdown("---")
    st.markdown("## 🔍 Data Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        completeness = (1 - summary['missing_values'] / (summary['total_records'] * summary['total_columns'])) * 100
        st.metric("Completeness", f"{completeness:.2f}%", 
                 delta="High Quality" if completeness > 95 else "Needs Attention")
    
    with col2:
        uniqueness = (1 - summary['duplicate_records'] / summary['total_records']) * 100
        st.metric("Uniqueness", f"{uniqueness:.2f}%",
                 delta="Excellent" if uniqueness > 99 else "Good")
    
    with col3:
        st.metric("Memory Usage", summary['memory_usage'])
    
    # Column information
    st.markdown("---")
    st.markdown("## 📋 Column Information")
    
    column_info = get_column_info(df)
    
    # Add search functionality
    search_term = st.text_input("🔍 Search columns:", "")
    if search_term:
        column_info = column_info[column_info['Column'].str.contains(search_term, case=False)]
    
    st.dataframe(column_info, use_container_width=True, height=400)
    
    # Missing values visualization
    st.markdown("---")
    st.markdown("## 🚨 Missing Values Analysis")
    
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ['Column', 'Missing_Count']
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        fig = px.bar(missing_df, x='Missing_Count', y='Column', orientation='h',
                    title='Missing Values by Column',
                    labels={'Missing_Count': 'Number of Missing Values', 'Column': 'Column Name'},
                    color='Missing_Count',
                    color_continuous_scale='Reds')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values found in the dataset!")
    
    # Data type distribution
    st.markdown("---")
    st.markdown("## 📊 Data Type Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dtype_counts = df.dtypes.value_counts().reset_index()
        dtype_counts.columns = ['Data Type', 'Count']
        # Convert dtype objects to strings for JSON serialization
        dtype_counts['Data Type'] = dtype_counts['Data Type'].astype(str)
        
        fig = px.pie(dtype_counts, values='Count', names='Data Type',
                    title='Column Data Types',
                    color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Numeric vs categorical split
        numeric_count = len(df.select_dtypes(include=['int64', 'float64']).columns)
        categorical_count = len(df.select_dtypes(include=['object']).columns)
        
        fig = go.Figure(data=[go.Pie(
            labels=['Numeric', 'Categorical'],
            values=[numeric_count, categorical_count],
            hole=0.4,
            marker_colors=['#1f77b4', '#ff7f0e']
        )])
        fig.update_layout(title='Numeric vs Categorical Features')
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample data preview
    st.markdown("---")
    st.markdown("## 👀 Data Preview")
    
    preview_tabs = st.tabs(["First 10 Rows", "Last 10 Rows", "Random Sample"])
    
    with preview_tabs[0]:
        st.dataframe(df.head(10), use_container_width=True)
    
    with preview_tabs[1]:
        st.dataframe(df.tail(10), use_container_width=True)
    
    with preview_tabs[2]:
        sample_size = st.slider("Select sample size:", 5, 50, 10)
        st.dataframe(df.sample(n=sample_size), use_container_width=True)
    
    # Statistical summary
    st.markdown("---")
    st.markdown("## 📊 Statistical Summary")
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if numeric_cols:
        selected_cols = st.multiselect(
            "Select columns for statistical summary:",
            numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        )
        
        if selected_cols:
            st.dataframe(df[selected_cols].describe().T, use_container_width=True)
    
    # Key insights
    st.markdown("---")
    st.markdown("## 💡 Key Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.info(f"""
        **Data Completeness:**
        - Total records: {summary['total_records']:,}
        - Complete records: {summary['total_records'] - summary['duplicate_records']:,}
        - Data quality: {'Excellent' if completeness > 95 else 'Good'}
        """)
    
    with insight_col2:
        st.success(f"""
        **Dataset Characteristics:**
        - Time span: 7 years (2017-2023)
        - Feature count: {summary['total_columns']}
        - Ready for modeling: ✅
        """)

else:
    st.error("Failed to load data. Please check the data file path.")
