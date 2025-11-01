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

### **Phase 2: Exploratory Data Analysis (EDA)** ✅
- **2.1 Univariate Analysis**: Distribution plots with statistical summaries
- **2.2 Temporal Analysis**: Monthly/yearly trends, seasonality patterns
- **2.3 Spatial Analysis**: State and district-level performance
- **2.4 System Comparison**: Automated vs Unautomated efficiency
- **2.5 Correlation Analysis**: Feature relationships and dependencies
- **2.6 Food Type Analysis**: Rice, Wheat, Coarse Grain, Fortified Rice patterns
- **2.7 Outlier Detection**: IQR-based anomaly identification
- **2.8 EDA Summary**: Key findings and insights for modeling

### **Phase 3: Feature Engineering** ✅
- **3.1 Derived Features**: 40+ engineered features
  - Automation rate, food type ratios, distribution gaps
  - Lag features (1-month, 3-month)
  - Rolling statistics (3-month moving averages)
  - Growth rates (month-over-month)
  - Geographic aggregates (state-level metrics)
- **3.2 Target Variables**: Multiple targets defined
  - Regression: Quantity & Efficiency prediction
  - Classification: Efficiency categories (Low/Medium/High)
- **3.3 Data Cleaning**: Outlier capping, missing value imputation
- **3.4 Categorical Encoding**: Label encoding for states, districts, systems
- **3.5 Feature Selection**: 50+ features for regression, 53+ for classification
- **3.6 Train-Test Split**: Time-based & random stratified splits
- **3.7 Feature Scaling**: StandardScaler & MinMaxScaler applied
- **3.8 Final Summary**: Complete feature engineering pipeline

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

- [x] **Project Setup** ✅
  - README created with comprehensive documentation
  - Notebook structure: `Food_Distribution_Analysis.ipynb`
  - Environment configuration (Plotly, scikit-learn, etc.)

- [x] **Phase 1: Data Understanding & Loading** ✅
  - Data loaded from `/kaggle/input/monthly-food-distribution-data/`
  - **63,425 records** × **34 columns** verified
  - Temporal coverage: **January 2017 - September 2023** (81 months)
  - Geographic coverage: **36 states**, **~700 districts**
  - Data quality: **Excellent** (minimal missing values)
  - Distribution system: **84.18% automated**, **15.82% unautomated**

- [x] **Phase 2: Exploratory Data Analysis (EDA)** ✅
  - ✅ **2.1 Univariate Analysis**: Distribution characteristics & statistical summaries
  - ✅ **2.2 Temporal Analysis**: 7-year trend analysis, seasonality detection
  - ✅ **2.3 Spatial Analysis**: State/district performance benchmarking
  - ✅ **2.4 System Comparison**: Automated vs Unautomated efficiency metrics
  - ✅ **2.5 Correlation Analysis**: Feature relationships heatmaps
  - ✅ **2.6 Food Type Analysis**: Rice (51%), Wheat (43%), Others (6%)
  - ✅ **2.7 Outlier Detection**: IQR method applied, anomalies identified
  - ✅ **2.8 EDA Summary**: Key insights documented for modeling
  
  **Key Findings**:
  - Overall distribution efficiency: **73.74%**
  - Automation growing steadily from 2017-2023
  - Top 5 states account for majority of distribution
  - Seasonal patterns: Minor variations, relatively stable
  - Strong correlation between allocated & distributed quantities

- [x] **Phase 3: Feature Engineering** ✅
  - ✅ **3.1 Derived Features**: 40+ new features created
    - Automation rate, food ratios, distribution gaps
    - Temporal: lag features, rolling stats, growth rates
    - Geographic: state aggregates, district rankings
  - ✅ **3.2 Target Variables**: 3 targets defined
    - Regression 1: `total_qty_distributed_epos`
    - Regression 2: `distribution_efficiency`
    - Classification: `efficiency_category` (Low/Medium/High)
  - ✅ **3.3 Data Cleaning**: Outliers capped, missing values handled
  - ✅ **3.4 Encoding**: States, districts, systems encoded
  - ✅ **3.5 Feature Selection**: 50+ regression, 53+ classification features
  - ✅ **3.6 Data Splitting**: Time-based & stratified splits (80-20)
  - ✅ **3.7 Scaling**: StandardScaler & MinMaxScaler fitted
  - ✅ **3.8 Final Summary**: Complete pipeline ready for modeling

- [ ] **Phase 4: Machine Learning Modeling (20-25 Models)** 🔄
  - [ ] Regression models (10-12 models)
  - [ ] Classification models (8-10 models)
  - [ ] Clustering & dimensionality reduction (2-3 models)
  - [ ] Model evaluation and comparison
  
- [ ] **Phase 5: Model Interpretation & Insights** 🔄
  - [ ] SHAP values analysis
  - [ ] Feature importance ranking
  - [ ] Visual storytelling with insights
  - [ ] Actionable recommendations
  
- [ ] **Final Report & Documentation** 🔄

---

## 👨‍💻 Author

**Data Science Project**  
Focus: End-to-end ML pipeline with emphasis on storytelling and insights

---

## 📝 Notes

This project emphasizes not just model accuracy, but **understanding the data story** — uncovering patterns, relationships, and actionable insights that can inform policy decisions about food distribution systems.

---

**Last Updated**: November 2, 2025  
**Status**: In Progress - Phase 3 Complete ✅ | Phase 4 Ready to Start �

---

## 🎉 Recent Updates

### November 2, 2025 - Phase 3 Complete! 🎊
- ✅ **Feature Engineering Pipeline Built** - Complete data preparation
- ✅ **40+ Derived Features Created**:
  - Automation metrics & food type ratios
  - Time-series features: lags, rolling stats, growth rates
  - Geographic aggregates: state-level averages & rankings
- ✅ **Multiple Target Variables Defined**:
  - 2 Regression targets (quantity & efficiency)
  - 1 Classification target (Low/Medium/High categories)
- ✅ **Data Cleaning & Transformation**:
  - Outliers capped at 150% efficiency
  - Missing values imputed intelligently
  - Categorical variables encoded (states, districts, systems)
- ✅ **Train-Test Splits Prepared**:
  - Time-based split for temporal integrity
  - Stratified split for balanced classification
  - 80-20 split ratio maintained
- ✅ **Feature Scaling Applied**:
  - StandardScaler for SVM & Neural Networks
  - MinMaxScaler as alternative
  - Original data preserved for tree-based models
- 📊 **Final Dataset**: ~50,000+ modeling-ready records with 50+ features

### November 2, 2025 - Phase 2 Complete! ✅
- ✅ **Comprehensive EDA completed** with 8 major analysis sections
- ✅ **Interactive & Static visualizations** using Plotly and Matplotlib/Seaborn
- ✅ **Key insights discovered**:
  - 84.18% of distribution through automated systems
  - Rice dominates at 51.18% of total allocation
  - Distribution efficiency averages 73.74%
  - Clear automation adoption trend over 2017-2023
  - Significant geographic variations in distribution patterns
  - Strong positive correlations between allocated & distributed quantities
- ✅ **Data Story Emerging**: Automation driving efficiency improvements across India's PDS
