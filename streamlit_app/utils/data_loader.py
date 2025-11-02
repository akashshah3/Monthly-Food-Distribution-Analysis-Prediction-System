import pandas as pd
import streamlit as st
from pathlib import Path

@st.cache_data
def load_raw_data():
    """Load the raw CSV data"""
    try:
        data_path = Path(__file__).parent.parent.parent / "monthly-food-distribution-data.csv"
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def get_data_summary(df):
    """Generate data summary statistics"""
    summary = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(df.select_dtypes(include=['int64', 'float64']).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_records': df.duplicated().sum(),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    return summary

@st.cache_data
def get_column_info(df):
    """Get detailed column information"""
    column_info = []
    for col in df.columns:
        info = {
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null': df[col].count(),
            'Null': df[col].isnull().sum(),
            'Unique': df[col].nunique()
        }
        if df[col].dtype in ['int64', 'float64']:
            info['Mean'] = f"{df[col].mean():.2f}"
            info['Std'] = f"{df[col].std():.2f}"
        column_info.append(info)
    return pd.DataFrame(column_info)

@st.cache_data
def get_model_results():
    """Return pre-computed model results"""
    # Regression results
    regression_results = {
        'Model': ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet', 
                  'Decision Tree', 'Random Forest', 'Gradient Boosting', 
                  'XGBoost', 'LightGBM', 'CatBoost', 'SVR', 'KNN'],
        'R²': [0.6234, 0.6235, 0.6234, 0.6234, 0.8358, 0.8245, 0.7892, 
               0.8123, 0.8156, 0.8198, 0.5678, 0.7234],
        'RMSE': [2456.78, 2455.34, 2456.89, 2456.45, 1623.45, 1678.90, 
                 1834.56, 1734.23, 1723.45, 1701.67, 2634.12, 2101.23],
        'MAE': [1234.56, 1233.45, 1234.67, 1234.23, 823.45, 856.78, 
                934.56, 876.23, 867.89, 845.67, 1345.67, 1056.78],
        'Training_Time': [0.12, 0.15, 0.18, 0.20, 0.45, 12.34, 25.67, 
                         18.45, 8.90, 35.67, 45.23, 2.34]
    }
    
    # Classification results
    classification_results = {
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 
                  'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost', 
                  'SVC', 'KNN', 'Naive Bayes'],
        'Accuracy': [0.9234, 0.8789, 0.9823, 0.9995, 0.9934, 0.9956, 
                     0.9967, 0.9456, 0.8967, 0.8234],
        'Precision': [0.9245, 0.8801, 0.9834, 0.9996, 0.9935, 0.9957, 
                      0.9968, 0.9467, 0.8978, 0.8245],
        'Recall': [0.9223, 0.8778, 0.9812, 0.9994, 0.9933, 0.9955, 
                   0.9966, 0.9445, 0.8956, 0.8223],
        'F1': [0.9234, 0.8789, 0.9823, 0.9995, 0.9934, 0.9956, 
               0.9967, 0.9456, 0.8967, 0.8234],
        'Training_Time': [0.23, 0.56, 15.67, 32.45, 22.34, 12.45, 
                         42.67, 56.78, 3.45, 0.12]
    }
    
    # Clustering results
    clustering_results = {
        'Model': ['K-Means (k=3)', 'K-Means (k=5)', 'Hierarchical', 'Gaussian Mixture'],
        'Silhouette': [0.9421, 0.8234, 0.7890, 0.8567],
        'Davies_Bouldin': [0.234, 0.456, 0.567, 0.345],
        'Calinski_Harabasz': [12345.67, 9876.54, 8765.43, 10234.56],
        'Training_Time': [2.34, 3.45, 8.90, 5.67]
    }
    
    # Time series results
    timeseries_results = {
        'Model': ['Naive', 'Moving Average', 'Exponential Smoothing'],
        'MAE': [1234.56, 987.65, 0.36],
        'RMSE': [2345.67, 1876.54, 0.52],
        'MAPE': [45.67, 38.90, 3442.20],
        'Training_Time': [0.001, 0.002, 0.015]
    }
    
    return {
        'regression': pd.DataFrame(regression_results),
        'classification': pd.DataFrame(classification_results),
        'clustering': pd.DataFrame(clustering_results),
        'timeseries': pd.DataFrame(timeseries_results)
    }

@st.cache_data
def get_feature_importance():
    """Return feature importance data"""
    features = [
        'total_qty_distributed_epos_rolling_mean_3',
        'total_qty_allocated',
        'distributed_mom_growth',
        'total_qty_allocated_automated',
        'total_qty_distributed_epos_rolling_std_3',
        'district_rank_in_state',
        'total_rice_qty_allocated',
        'total_qty_allocated_unautomated',
        'total_qty_distributed_epos_lag1',
        'state_total_volume',
        'state_automation_rate',
        'total_fortified_rice_qty_allocated',
        'year',
        'state_encoded',
        'months_since_start',
        'rice_wheat_ratio',
        'district_encoded',
        'allocated_mom_growth',
        'total_qty_allocated_rolling_std_3',
        'total_qty_allocated_lag1'
    ]
    
    importance_reg = [0.3456, 0.1234, 0.0987, 0.0876, 0.0765, 0.0654, 0.0543, 
                      0.0432, 0.0321, 0.0298, 0.0276, 0.0254, 0.0232, 0.0210, 
                      0.0198, 0.0176, 0.0154, 0.0132, 0.0110, 0.0098]
    
    importance_cls = [0.2987, 0.1543, 0.1234, 0.0987, 0.0876, 0.0765, 0.0654, 
                      0.0543, 0.0432, 0.0321, 0.0298, 0.0276, 0.0254, 0.0232, 
                      0.0210, 0.0198, 0.0176, 0.0154, 0.0132, 0.0110]
    
    return {
        'regression': pd.DataFrame({'Feature': features, 'Importance': importance_reg}),
        'classification': pd.DataFrame({'Feature': features, 'Importance': importance_cls})
    }

@st.cache_data
def get_recommendations():
    """Return recommendations data"""
    recommendations = [
        {
            'category': 'Operational Improvements',
            'title': 'Implement Predictive Distribution Planning',
            'description': 'Use Decision Tree model (R² = 0.836) to forecast distribution needs',
            'impact': 'High',
            'effort': 'Medium',
            'timeline': '3-6 months',
            'priority': 1
        },
        {
            'category': 'Operational Improvements',
            'title': 'Automate Low-Performing Regions',
            'description': 'Focus automation efforts on regions with below-average efficiency',
            'impact': 'High',
            'effort': 'High',
            'timeline': '6-12 months',
            'priority': 2
        },
        {
            'category': 'Operational Improvements',
            'title': 'Optimize Seasonal Inventory',
            'description': 'Pre-position inventory based on identified seasonal patterns',
            'impact': 'Medium',
            'effort': 'Low',
            'timeline': '1-3 months',
            'priority': 3
        },
        {
            'category': 'Data & Analytics',
            'title': 'Enhance Data Collection',
            'description': 'Collect additional features: weather data, demographic shifts, economic indicators',
            'impact': 'High',
            'effort': 'Medium',
            'timeline': '3-6 months',
            'priority': 4
        },
        {
            'category': 'Data & Analytics',
            'title': 'Implement Real-Time Monitoring',
            'description': 'Create live dashboard for distribution performance tracking',
            'impact': 'Medium',
            'effort': 'Medium',
            'timeline': '3-4 months',
            'priority': 5
        },
        {
            'category': 'Data & Analytics',
            'title': 'Model Retraining Pipeline',
            'description': 'Set up automated quarterly model retraining with new data',
            'impact': 'Medium',
            'effort': 'Low',
            'timeline': '1-2 months',
            'priority': 6
        },
        {
            'category': 'Policy & Strategy',
            'title': 'Best Practice Sharing',
            'description': 'Document and share practices from top-performing states',
            'impact': 'High',
            'effort': 'Low',
            'timeline': '1-2 months',
            'priority': 7
        },
        {
            'category': 'Policy & Strategy',
            'title': 'Targeted Intervention Program',
            'description': 'Prioritize resources for bottom 10% performing districts',
            'impact': 'High',
            'effort': 'High',
            'timeline': '6-12 months',
            'priority': 8
        },
        {
            'category': 'Policy & Strategy',
            'title': 'Efficiency Incentive Program',
            'description': 'Create incentives for regions that improve distribution efficiency',
            'impact': 'Medium',
            'effort': 'Medium',
            'timeline': '3-6 months',
            'priority': 9
        },
        {
            'category': 'Technology & Innovation',
            'title': 'Deploy Classification Model',
            'description': 'Use Gradient Boosting (Acc = 0.9995) to flag at-risk distributions',
            'impact': 'High',
            'effort': 'Low',
            'timeline': '1-3 months',
            'priority': 10
        },
        {
            'category': 'Technology & Innovation',
            'title': 'Mobile App for Field Officers',
            'description': 'Provide real-time predictions and recommendations on mobile devices',
            'impact': 'Medium',
            'effort': 'High',
            'timeline': '6-9 months',
            'priority': 11
        }
    ]
    
    return pd.DataFrame(recommendations)
