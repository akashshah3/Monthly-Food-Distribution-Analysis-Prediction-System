import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import load_raw_data

st.set_page_config(page_title="Exploratory Analysis", page_icon="📈", layout="wide")

st.title("📈 Exploratory Data Analysis")
st.markdown("### Interactive visualizations and statistical insights")

# Load data
df = load_raw_data()

if df is not None:
    # Sidebar filters
    with st.sidebar:
        st.markdown("### 🎛️ Filters")
        
        # Year filter
        if 'year' in df.columns:
            years = sorted(df['year'].unique())
            selected_years = st.multiselect("Select Years:", years, default=years)
            df_filtered = df[df['year'].isin(selected_years)]
        else:
            df_filtered = df
        
        st.metric("Filtered Records", f"{len(df_filtered):,}")
    
    # Distribution Analysis
    st.markdown("---")
    st.markdown("## 📊 Distribution Analysis")
    
    numeric_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if numeric_cols:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_col = st.selectbox("Select column to visualize:", numeric_cols)
            chart_type = st.radio("Chart type:", ["Histogram", "Box Plot", "Violin Plot"])
        
        with col2:
            if chart_type == "Histogram":
                fig = px.histogram(df_filtered, x=selected_col,
                                  title=f'Distribution of {selected_col}',
                                  color_discrete_sequence=['#1f77b4'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Box Plot":
                fig = px.box(df_filtered, y=selected_col,
                            title=f'Box Plot of {selected_col}',
                            color_discrete_sequence=['#1f77b4'])
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                fig = px.violin(df_filtered, y=selected_col,
                               title=f'Violin Plot of {selected_col}',
                               color_discrete_sequence=['#1f77b4'])
                st.plotly_chart(fig, use_container_width=True)
    
    # Temporal Analysis
    st.markdown("---")
    st.markdown("## 📅 Temporal Trends")
    
    if 'year' in df_filtered.columns and 'month' in df_filtered.columns:
        # Create date column
        df_temporal = df_filtered.copy()
        
        # Aggregate by year-month
        if 'total_qty_allocated_epos' in df_temporal.columns:
            temporal_agg = df_temporal.groupby(['year', 'month'])['total_qty_allocated_epos'].sum().reset_index()
            temporal_agg['date'] = pd.to_datetime(temporal_agg[['year', 'month']].assign(day=1))
            
            fig = px.line(temporal_agg, x='date', y='total_qty_allocated_epos',
                         title='Total Quantity Distributed Over Time',
                         labels={'total_qty_allocated_epos': 'Total Quantity', 'date': 'Date'})
            fig.update_traces(line_color='#1f77b4', line_width=2)
            st.plotly_chart(fig, use_container_width=True)
            
            # Year-over-Year comparison
            yearly_totals = df_temporal.groupby('year')['total_qty_allocated_epos'].sum().reset_index()
            
            fig = px.bar(yearly_totals, x='year', y='total_qty_allocated_epos',
                        title='Year-over-Year Distribution',
                        labels={'total_qty_allocated_epos': 'Total Quantity', 'year': 'Year'},
                        color='total_qty_allocated_epos',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis
    st.markdown("---")
    st.markdown("## 🔗 Correlation Analysis")
    
    if len(numeric_cols) > 1:
        corr_cols = st.multiselect(
            "Select columns for correlation analysis:",
            numeric_cols,
            default=numeric_cols[:8] if len(numeric_cols) > 8 else numeric_cols
        )
        
        if len(corr_cols) > 1:
            corr_matrix = df_filtered[corr_cols].corr()
            
            fig = px.imshow(corr_matrix,
                           text_auto='.2f',
                           aspect='auto',
                           color_continuous_scale='RdBu_r',
                           title='Correlation Heatmap')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations
            st.markdown("### 🔝 Strongest Correlations")
            
            # Get upper triangle
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False).head(10)
            
            st.dataframe(corr_df, use_container_width=True)
    
    # Categorical Analysis
    st.markdown("---")
    st.markdown("## 🏷️ Categorical Features Analysis")
    
    cat_cols = df_filtered.select_dtypes(include=['object']).columns.tolist()
    
    if cat_cols:
        selected_cat = st.selectbox("Select categorical column:", cat_cols)
        
        if selected_cat:
            value_counts = df_filtered[selected_cat].value_counts().head(15)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                            title=f'Top 15 Values in {selected_cat}',
                            labels={'x': selected_cat, 'y': 'Count'},
                            color=value_counts.values,
                            color_continuous_scale='Blues')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                            title=f'Distribution of {selected_cat} (Top 15)',
                            hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.info(f"""
            **{selected_cat} Statistics:**
            - Unique values: {df_filtered[selected_cat].nunique():,}
            - Most common: {value_counts.index[0]} ({value_counts.values[0]:,} occurrences)
            - Least common (in top 15): {value_counts.index[-1]} ({value_counts.values[-1]:,} occurrences)
            """)
    
    # Scatter Plot Analysis
    st.markdown("---")
    st.markdown("## 🎯 Relationship Explorer")
    
    if len(numeric_cols) >= 2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("X-axis:", numeric_cols, index=0)
        with col2:
            y_axis = st.selectbox("Y-axis:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        with col3:
            color_by = st.selectbox("Color by:", ['None'] + cat_cols)
        
        if x_axis and y_axis:
            if color_by != 'None':
                fig = px.scatter(df_filtered, x=x_axis, y=y_axis, color=color_by,
                                title=f'{y_axis} vs {x_axis}',
                                opacity=0.6)
            else:
                fig = px.scatter(df_filtered, x=x_axis, y=y_axis,
                                title=f'{y_axis} vs {x_axis}',
                                opacity=0.6,
                                color_discrete_sequence=['#1f77b4'])
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Key Statistics
    st.markdown("---")
    st.markdown("## 📊 Key Statistics Summary")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Total Records", f"{len(df_filtered):,}")
    
    with stat_col2:
        if 'total_qty_allocated_epos' in df_filtered.columns:
            total_qty = df_filtered['total_qty_allocated_epos'].sum()
            st.metric("Total Distribution", f"{total_qty/1e6:.2f}M units")
    
    with stat_col3:
        if 'year' in df_filtered.columns:
            year_range = f"{df_filtered['year'].min()}-{df_filtered['year'].max()}"
            st.metric("Time Period", year_range)
    
    with stat_col4:
        if cat_cols:
            st.metric("Categorical Features", len(cat_cols))

else:
    st.error("Failed to load data.")
