# 📊 Comprehensive EDA and Predictive Modeling on Monthly Food Distribution Data

## 🎯 Project Overview

This project performs an **end-to-end data science analysis** on India's Public Distribution System (PDS) monthly food distribution data. The analysis covers **comprehensive exploratory data analysis (EDA)**, **feature engineering**, and **20-25 machine learning models** to uncover insights about food distribution patterns, efficiency, and trends across different states and districts.

---

## 📁 Dataset Information

**Dataset**: `monthly-food-distribution-data.csv`

**Size**: 63,427 records

**Time Period**: 2017 onwards (Monthly data)

**Geographic Coverage**: Multiple Indian states and districts

### Key Features:
- **Temporal**: Monthly timestamps
- **Geographic**: State names, codes, district names, codes
- **Food Types**: Rice, Wheat, Coarse Grain, Fortified Rice
- **Distribution Channels**: Automated and Unautomated systems
- **Metrics**: Allocation quantities, distribution quantities, distribution percentages

---

## 🎯 Project Objectives

### 1. **Multiple Target Variables**
- **Regression Target 1**: `total_qty_distributed_epos` (Total quantity distributed)
- **Regression Target 2**: `percentage_qty_distributed_automated` (Automation efficiency)
- **Classification Target**: Distribution efficiency categories (High/Medium/Low)

### 2. **Comprehensive Analysis Focus**
- ✅ **Temporal Analysis**: Trends, seasonality, year-over-year patterns
- ✅ **Spatial Analysis**: State and district-level distribution patterns
- ✅ **System Comparison**: Automated vs Unautomated distribution efficiency
- ✅ **Food Type Analysis**: Rice, wheat, coarse grain, and fortified rice patterns

### 3. **Machine Learning Models (20-25+)**
- **Linear Models**: Linear Regression, Ridge, Lasso, ElasticNet
- **Tree-Based Models**: Decision Tree, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, Gradient Boosting
- **Support Vector Machines**: SVR, SVC
- **Instance-Based**: KNN Regressor, KNN Classifier
- **Neural Networks**: MLP Regressor, MLP Classifier
- **Ensemble**: Voting, Stacking, Bagging, AdaBoost
- **Clustering**: KMeans, DBSCAN, Hierarchical
- **Dimensionality Reduction**: PCA, t-SNE
- **Time Series**: ARIMA, Prophet, LSTM (if applicable)

---

## 📊 Project Structure

```
DS_Project/
│
├── monthly-food-distribution-data.csv          # Raw dataset
├── Food_Distribution_Analysis.ipynb             # Main analysis notebook
├── AGENTS.md                                    # Project guidelines
├── README.md                                    # This file
│
└── (Future additions)
    ├── models/                                  # Saved models
    ├── visualizations/                          # Generated plots
    └── reports/                                 # Analysis reports
```

---

## 🚀 Analysis Pipeline

### **Phase 1: Data Understanding & Loading** ⏳
- Load and examine dataset structure
- Data types, missing values, duplicates analysis
- Statistical summary

### **Phase 2: Exploratory Data Analysis (EDA)** 🔄
- Univariate analysis
- Temporal trends and seasonality
- Spatial distribution patterns
- Correlation analysis
- Anomaly detection

### **Phase 3: Feature Engineering** 🔄
- Derived metrics creation
- Encoding categorical variables
- Feature scaling and normalization
- Target variable creation

### **Phase 4: Machine Learning Modeling** 🔄
- Train-test split
- Model training (20-25 models)
- Hyperparameter tuning
- Model evaluation and comparison

### **Phase 5: Model Interpretation & Insights** 🔄
- Feature importance analysis
- SHAP value interpretation
- Visual storytelling
- Actionable recommendations

---

## 📈 Key Questions to Answer

1. **Temporal Insights**:
   - Are there seasonal patterns in food distribution?
   - How has distribution evolved over time?
   - Can we forecast future distribution needs?

2. **Spatial Insights**:
   - Which states/districts receive the most food grain?
   - Are there regional disparities in distribution?
   - Which regions show highest/lowest efficiency?

3. **System Efficiency**:
   - How does automated vs unautomated distribution compare?
   - What factors drive distribution efficiency?
   - Which system shows better performance?

4. **Predictive Insights**:
   - Can we predict distribution quantities accurately?
   - What features are most important for predictions?
   - Which model performs best for this data?

---

## 🛠️ Technologies & Libraries Used

### **Data Processing**
- `pandas` - Data manipulation
- `numpy` - Numerical operations

### **Visualization**
- `matplotlib` - Static visualizations
- `seaborn` - Statistical visualizations
- `plotly` - Interactive visualizations

### **Machine Learning**
- `scikit-learn` - Classical ML algorithms
- `xgboost` - Gradient boosting
- `lightgbm` - Light gradient boosting
- `catboost` - Categorical boosting
- `tensorflow/keras` - Deep learning (if applicable)

### **Model Interpretation**
- `shap` - SHAP values for explainability
- `lime` - Local interpretability

### **Statistical Analysis**
- `scipy` - Statistical tests
- `statsmodels` - Time series analysis

---

## 📊 Current Progress

- [x] Project setup and README creation
- [x] Notebook created: `Food_Distribution_Analysis.ipynb`
- [x] Phase 1: Data Understanding & Loading ✅
  - Data loaded from `/kaggle/input/monthly-food-distribution-data/`
  - 63,425 records × 34 columns verified
  - Temporal features extracted (2017-2023)
  - Data quality assessed (excellent quality)
- [x] Phase 2: Exploratory Data Analysis ✅
  - ✅ Section 2.1: Univariate Analysis (Distribution plots)
  - ✅ Section 2.2: Temporal & Seasonal Analysis
  - ✅ Section 2.3: Spatial Analysis (State & District patterns)
  - ✅ Section 2.4: Automated vs Unautomated Comparison
  - ✅ Section 2.5: Correlation Analysis
  - ✅ Section 2.6: Food Type Analysis
  - ✅ Section 2.7: Outlier Detection
  - ✅ Section 2.8: EDA Summary & Key Insights
- [ ] Phase 3: Feature Engineering 🔄
- [ ] Phase 4: Machine Learning Modeling (20-25 Models)
- [ ] Phase 5: Model Interpretation & Insights
- [ ] Final Report & Documentation

---

## 👨‍💻 Author

**Data Science Project**  
Focus: End-to-end ML pipeline with emphasis on storytelling and insights

---

## 📝 Notes

This project emphasizes not just model accuracy, but **understanding the data story** — uncovering patterns, relationships, and actionable insights that can inform policy decisions about food distribution systems.

---

**Last Updated**: November 2, 2025  
**Status**: In Progress - Phase 2 Complete ✅ | Phase 3 Starting 🔄

---

## 🎉 Recent Updates

### November 2, 2025 - Phase 2 Complete!
- ✅ **Comprehensive EDA completed** with 8 major analysis sections
- ✅ **Interactive visualizations** using Plotly and Matplotlib/Seaborn
- ✅ **Key insights discovered**:
  - 84.18% of distribution through automated systems
  - Rice dominates at 51.18% of total allocation
  - Distribution efficiency averages 73.74%
  - Clear automation adoption trend over 2017-2023
  - Significant geographic variations in distribution patterns
