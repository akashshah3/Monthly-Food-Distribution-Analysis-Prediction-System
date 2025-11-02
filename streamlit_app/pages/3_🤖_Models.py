import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import get_model_results

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Model Performance")
st.markdown("### Comprehensive analysis of 29 trained models")

# Get model results
model_results = get_model_results()

# Model category selector
st.markdown("---")
model_category = st.selectbox(
    "📂 Select Model Category:",
    ["Overview", "Regression Models", "Classification Models", "Clustering Models", "Time Series Models"]
)

if model_category == "Overview":
    st.markdown("## 📊 Model Performance Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", "29", delta="All Trained")
    with col2:
        st.metric("Best R² Score", "0.8358", delta="Regression")
    with col3:
        st.metric("Best Accuracy", "99.95%", delta="Classification")
    with col4:
        st.metric("Best Silhouette", "0.9421", delta="Clustering")
    
    # Model counts by category
    st.markdown("---")
    st.markdown("### 📈 Models by Category")
    
    category_data = pd.DataFrame({
        'Category': ['Regression', 'Classification', 'Clustering', 'Time Series'],
        'Count': [12, 10, 4, 3],
        'Best Model': ['Decision Tree', 'Gradient Boosting', 'K-Means (k=3)', 'Exponential Smoothing']
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(category_data, x='Category', y='Count',
                    title='Number of Models by Category',
                    color='Count',
                    color_continuous_scale='Blues',
                    text='Count')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(category_data, values='Count', names='Category',
                    title='Model Distribution',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig, use_container_width=True)
    
    # Best models summary
    st.markdown("---")
    st.markdown("### 🏆 Best Models by Category")
    
    best_models = pd.DataFrame({
        'Category': ['Regression', 'Classification', 'Clustering', 'Time Series'],
        'Best Model': ['Decision Tree', 'Gradient Boosting', 'K-Means (k=3)', 'Exponential Smoothing'],
        'Primary Metric': ['R² = 0.8358', 'Accuracy = 99.95%', 'Silhouette = 0.9421', 'MAE = 0.36'],
        'Training Time': ['0.45s', '32.45s', '2.34s', '0.015s']
    })
    
    st.dataframe(best_models, use_container_width=True, hide_index=True)
    
    # Training time comparison
    st.markdown("---")
    st.markdown("### ⏱️ Training Time Comparison")
    
    all_models = []
    for category, df in model_results.items():
        df_temp = df.copy()
        df_temp['Category'] = category.title()
        all_models.append(df_temp)
    
    combined_df = pd.concat(all_models, ignore_index=True)
    
    fig = px.bar(combined_df, x='Model', y='Training_Time',
                color='Category',
                title='Training Time by Model',
                labels={'Training_Time': 'Training Time (seconds)'},
                barmode='group')
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

elif model_category == "Regression Models":
    st.markdown("## 📈 Regression Model Performance")
    
    df_reg = model_results['regression']
    
    # Best model highlight
    best_idx = df_reg['R²'].idxmax()
    best_model = df_reg.loc[best_idx]
    
    st.success(f"""
    ### 🏆 Best Regression Model: **{best_model['Model']}**
    - **R² Score:** {best_model['R²']:.4f}
    - **RMSE:** {best_model['RMSE']:.2f}
    - **MAE:** {best_model['MAE']:.2f}
    - **Training Time:** {best_model['Training_Time']:.2f}s
    """)
    
    # Metrics comparison
    st.markdown("---")
    st.markdown("### 📊 Model Comparison")
    
    tab1, tab2, tab3 = st.tabs(["R² Score", "RMSE", "MAE"])
    
    with tab1:
        fig = px.bar(df_reg.sort_values('R²', ascending=True), 
                    x='R²', y='Model', orientation='h',
                    title='R² Score Comparison (Higher is Better)',
                    color='R²',
                    color_continuous_scale='Blues',
                    text='R²')
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.bar(df_reg.sort_values('RMSE', ascending=False), 
                    x='RMSE', y='Model', orientation='h',
                    title='RMSE Comparison (Lower is Better)',
                    color='RMSE',
                    color_continuous_scale='Reds_r',
                    text='RMSE')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.bar(df_reg.sort_values('MAE', ascending=False), 
                    x='MAE', y='Model', orientation='h',
                    title='MAE Comparison (Lower is Better)',
                    color='MAE',
                    color_continuous_scale='Oranges_r',
                    text='MAE')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.markdown("---")
    st.markdown("### 📋 Detailed Metrics")
    st.dataframe(df_reg.sort_values('R²', ascending=False), use_container_width=True, hide_index=True)

elif model_category == "Classification Models":
    st.markdown("## 🎯 Classification Model Performance")
    
    df_cls = model_results['classification']
    
    # Best model highlight
    best_idx = df_cls['Accuracy'].idxmax()
    best_model = df_cls.loc[best_idx]
    
    st.success(f"""
    ### 🏆 Best Classification Model: **{best_model['Model']}**
    - **Accuracy:** {best_model['Accuracy']:.4f} ({best_model['Accuracy']*100:.2f}%)
    - **Precision:** {best_model['Precision']:.4f}
    - **Recall:** {best_model['Recall']:.4f}
    - **F1 Score:** {best_model['F1']:.4f}
    - **Training Time:** {best_model['Training_Time']:.2f}s
    """)
    
    # Metrics comparison
    st.markdown("---")
    st.markdown("### 📊 Model Comparison")
    
    # Multi-metric comparison
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Accuracy', x=df_cls['Model'], y=df_cls['Accuracy']))
    fig.add_trace(go.Bar(name='Precision', x=df_cls['Model'], y=df_cls['Precision']))
    fig.add_trace(go.Bar(name='Recall', x=df_cls['Model'], y=df_cls['Recall']))
    fig.add_trace(go.Bar(name='F1', x=df_cls['Model'], y=df_cls['F1']))
    
    fig.update_layout(
        title='Classification Metrics Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=500,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Accuracy comparison
    st.markdown("---")
    fig = px.bar(df_cls.sort_values('Accuracy', ascending=True), 
                x='Accuracy', y='Model', orientation='h',
                title='Accuracy Comparison',
                color='Accuracy',
                color_continuous_scale='Greens',
                text='Accuracy')
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.markdown("---")
    st.markdown("### 📋 Detailed Metrics")
    st.dataframe(df_cls.sort_values('Accuracy', ascending=False), use_container_width=True, hide_index=True)

elif model_category == "Clustering Models":
    st.markdown("## 🔵 Clustering Model Performance")
    
    df_cluster = model_results['clustering']
    
    # Best model highlight
    best_idx = df_cluster['Silhouette'].idxmax()
    best_model = df_cluster.loc[best_idx]
    
    st.success(f"""
    ### 🏆 Best Clustering Model: **{best_model['Model']}**
    - **Silhouette Score:** {best_model['Silhouette']:.4f}
    - **Davies-Bouldin Index:** {best_model['Davies_Bouldin']:.3f}
    - **Calinski-Harabasz Score:** {best_model['Calinski_Harabasz']:.2f}
    - **Training Time:** {best_model['Training_Time']:.2f}s
    """)
    
    # Metrics comparison
    st.markdown("---")
    st.markdown("### 📊 Clustering Metrics Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(df_cluster, x='Model', y='Silhouette',
                    title='Silhouette Score (Higher is Better)',
                    color='Silhouette',
                    color_continuous_scale='Blues',
                    text='Silhouette')
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(df_cluster, x='Model', y='Davies_Bouldin',
                    title='Davies-Bouldin Index (Lower is Better)',
                    color='Davies_Bouldin',
                    color_continuous_scale='Reds_r',
                    text='Davies_Bouldin')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.markdown("---")
    st.markdown("### 📋 Detailed Metrics")
    st.dataframe(df_cluster.sort_values('Silhouette', ascending=False), use_container_width=True, hide_index=True)

elif model_category == "Time Series Models":
    st.markdown("## 📅 Time Series Model Performance")
    
    df_ts = model_results['timeseries']
    
    # Best model highlight
    best_idx = df_ts['MAE'].idxmin()
    best_model = df_ts.loc[best_idx]
    
    st.success(f"""
    ### 🏆 Best Time Series Model: **{best_model['Model']}**
    - **MAE:** {best_model['MAE']:.2f}
    - **RMSE:** {best_model['RMSE']:.2f}
    - **MAPE:** {best_model['MAPE']:.2f}%
    - **Training Time:** {best_model['Training_Time']:.3f}s
    """)
    
    # Metrics comparison
    st.markdown("---")
    st.markdown("### 📊 Forecasting Accuracy Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(df_ts, x='Model', y='MAE',
                    title='MAE Comparison (Lower is Better)',
                    color='MAE',
                    color_continuous_scale='Blues_r',
                    text='MAE')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(df_ts, x='Model', y='RMSE',
                    title='RMSE Comparison (Lower is Better)',
                    color='RMSE',
                    color_continuous_scale='Reds_r',
                    text='RMSE')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.markdown("---")
    st.markdown("### 📋 Detailed Metrics")
    st.dataframe(df_ts.sort_values('MAE'), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.info("💡 **Tip:** All models were trained on 80% of the data and evaluated on the remaining 20% test set.")
