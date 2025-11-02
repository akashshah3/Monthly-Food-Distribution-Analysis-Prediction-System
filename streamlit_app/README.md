# 📊 Food Distribution Analysis Dashboard

A comprehensive Streamlit dashboard for presenting data science analysis of monthly food distribution data across India (2019-2021).

## 🎯 Project Overview

This dashboard presents the complete end-to-end analysis including:
- **Data Quality & Exploration**: 58K+ records, 284M+ units distributed
- **Exploratory Data Analysis**: Interactive visualizations and insights
- **Machine Learning Models**: 29 models across 4 categories (regression, classification, clustering, time series)
- **Model Interpretability**: Feature importance, SHAP analysis, error analysis
- **Business Impact**: ROI projections showing $1.5M 3-year return
- **Strategic Recommendations**: 11 actionable recommendations with priority matrix

## 🚀 Quick Start

### Local Development

1. **Clone or navigate to the project directory**
```bash
cd /home/akash/Organization/01_Projects/DS_Project/streamlit_app
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the dashboard**
```bash
streamlit run app.py
```

4. **Open your browser**
Navigate to: `http://localhost:8501`

### Data Requirements

Ensure the dataset is available at:
```
../monthly-food-distribution-data.csv
```

## 📁 Project Structure

```
streamlit_app/
├── app.py                           # Main landing page
├── requirements.txt                 # Python dependencies
├── .streamlit/
│   └── config.toml                  # Streamlit configuration
├── pages/
│   ├── 1_📊_Data_Overview.py       # Dataset exploration
│   ├── 2_📈_EDA.py                 # Exploratory analysis
│   ├── 3_🤖_Models.py              # Model performance
│   ├── 4_🔍_Insights.py            # Feature importance & SHAP
│   ├── 5_💼_Business_Impact.py     # ROI & projections
│   └── 6_🎯_Recommendations.py     # Action plan
└── utils/
    └── data_loader.py               # Data loading utilities
```

## 📄 Pages Overview

### 🏠 Home
- Project overview and key metrics
- Analysis phases and achievements
- Model portfolio summary
- Quick navigation guide

### 📊 Data Overview
- Dataset summary (58K records, 38 features)
- Data quality metrics (99.4% completeness)
- Column information and statistics
- Missing values analysis
- Sample data preview

### 📈 Exploratory Data Analysis
- Distribution analysis with multiple chart types
- Temporal trends over time
- Correlation heatmap
- Categorical feature analysis
- Interactive relationship explorer
- Year-based filtering

### 🤖 Models
- **Overview**: 29 models trained
- **Regression**: 12 models (Best: Decision Tree, R²=0.8358)
- **Classification**: 10 models (Best: Gradient Boosting, 99.95% accuracy)
- **Clustering**: 4 models (Best: DBSCAN, Silhouette=0.9421)
- **Time Series**: 3 models (Best: XGBoost, MAE=1702.86)

### 🔍 Model Insights
- Feature importance analysis
- SHAP value interpretations
- Actual vs predicted visualizations
- Residual analysis
- Error analysis and patterns
- Model limitations

### 💼 Business Impact
- Current operations overview
- ROI calculator with adjustable assumptions
- 5-year projections
- Optimization scenarios (Conservative/Moderate/Aggressive)
- Cost-benefit analysis
- Investment justification

### 🎯 Recommendations
- 11 prioritized recommendations
- Impact vs Effort priority matrix
- Phase-based implementation plan
- Resource allocation estimates
- Category and priority filtering
- Interactive checklist

## 🔧 Key Features

- **Interactive Visualizations**: Built with Plotly for rich interactivity
- **Data Caching**: Optimized performance with Streamlit caching
- **Responsive Design**: Works on desktop and tablet devices
- **Pre-computed Results**: Fast loading with cached model results
- **Filter & Drill-down**: Multiple filtering options across pages
- **Professional Styling**: Custom CSS and consistent color scheme

## 📊 Model Results Summary

| Category | Best Model | Key Metric | Value |
|----------|-----------|------------|-------|
| Regression | Decision Tree | R² Score | 0.8358 |
| Classification | Gradient Boosting | Accuracy | 99.95% |
| Clustering | DBSCAN | Silhouette | 0.9421 |
| Time Series | XGBoost | MAE | 1702.86 |

## 💡 Technical Stack

- **Frontend**: Streamlit 1.31.0
- **Data Processing**: Pandas 2.2.3
- **Visualizations**: Plotly 5.18.0
- **Numerical Computing**: NumPy 1.26.3
- **Python**: 3.11+

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file as `app.py`
5. Deploy!

**Note**: Ensure `monthly-food-distribution-data.csv` is in the parent directory or update the path in `utils/data_loader.py`.

### Alternative Deployment Options

- **Docker**: Containerize the application
- **Heroku**: Deploy with Procfile
- **AWS/GCP/Azure**: Cloud platform deployment
- **Local Network**: Run on internal network

## 📈 Performance Optimization

The dashboard uses several optimization techniques:
- `@st.cache_data` for data loading
- Pre-computed model results (no live inference)
- Efficient data structures
- Lazy loading of visualizations
- Minimal external API calls

## 🎓 University Project Context

This dashboard was created for a university data science project to demonstrate:
- End-to-end data science workflow
- Advanced analytics techniques
- Business value communication
- Professional presentation skills
- Technical and non-technical audience engagement

## 🤝 Contributing

This is a university project. For questions or suggestions:
- Review the Jupyter notebook: `Food_Distribution_Analysis.ipynb`
- Check the analysis results in the notebook cells
- Refer to the dashboard documentation

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Dataset: Monthly Food Distribution Data (2019-2021)
- Analysis Period: 3 years across multiple Indian states
- Total Records: 58,016 modeling-ready records
- Total Distribution: 284M+ food units

## 📞 Support

For issues or questions about the dashboard:
1. Check the README
2. Review the Jupyter notebook analysis
3. Examine the code comments in each page
4. Verify data file paths

---

**Built with ❤️ using Streamlit**

*Last Updated: 2024*
