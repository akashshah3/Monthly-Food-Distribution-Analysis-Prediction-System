import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Business Impact", page_icon="💼", layout="wide")

st.title("💼 Business Impact & ROI Analysis")
st.markdown("### Quantifying the value of predictive analytics")

# Tab selection
tab1, tab2, tab3, tab4 = st.tabs(["📊 Current State", "💰 ROI Analysis", "🎯 Optimization", "📈 Projections"])

with tab1:
    st.markdown("## 📊 Current Operations Overview")
    
    # Current state metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Distribution", "284.2M units", delta="Annual")
    with col2:
        st.metric("Average per Record", "4,899 units")
    with col3:
        st.metric("Distribution Efficiency", "85.3%", delta="+5.2%")
    with col4:
        st.metric("Waste Percentage", "14.7%", delta="-2.1%", delta_color="inverse")
    
    st.markdown("---")
    
    # Current state breakdown
    st.markdown("### 📋 Current Operations Breakdown")
    
    current_state = pd.DataFrame({
        'Metric': [
            'Total Records',
            'Food Units Distributed',
            'Average per Transaction',
            'Distribution Centers',
            'Automation Rate',
            'Manual Processing Time',
            'Average Cost per Transaction'
        ],
        'Value': [
            '58,016',
            '284,173,812',
            '4,899',
            '---',
            '---',
            '---',
            '$12.50'
        ],
        'Notes': [
            '2019-2021 period',
            'Total quantity allocated',
            'Mean distribution size',
            'Multiple states/districts',
            'Currently low',
            'High manual effort',
            'Includes processing overhead'
        ]
    })
    
    st.dataframe(current_state, use_container_width=True, hide_index=True)
    
    # Visual breakdown
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        efficiency_data = pd.DataFrame({
            'Category': ['Efficient', 'Inefficient', 'Waste'],
            'Percentage': [70.3, 15.0, 14.7]
        })
        
        fig = px.pie(efficiency_data, values='Percentage', names='Category',
                    title='Current Distribution Efficiency',
                    color='Category',
                    color_discrete_map={'Efficient': 'green', 'Inefficient': 'orange', 'Waste': 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        process_data = pd.DataFrame({
            'Process': ['Manual', 'Semi-Automated', 'Automated'],
            'Percentage': [65, 25, 10]
        })
        
        fig = px.bar(process_data, x='Process', y='Percentage',
                    title='Current Process Distribution',
                    color='Process',
                    text='Percentage')
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("## 💰 Return on Investment Analysis")
    
    # ROI Calculator
    st.markdown("### 🧮 ROI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💵 Cost Assumptions")
        
        implementation_cost = st.number_input("Implementation Cost ($)", 
                                             value=150000, step=10000, 
                                             help="One-time setup and deployment cost")
        annual_maintenance = st.number_input("Annual Maintenance ($)", 
                                            value=30000, step=5000,
                                            help="Yearly maintenance and support")
        cost_per_transaction = st.slider("Current Cost per Transaction ($)", 
                                        5.0, 20.0, 12.5, 0.5)
    
    with col2:
        st.markdown("#### 📈 Benefit Assumptions")
        
        efficiency_gain = st.slider("Efficiency Gain (%)", 
                                    5, 30, 15, 1,
                                    help="Expected improvement in distribution efficiency")
        waste_reduction = st.slider("Waste Reduction (%)", 
                                    5, 40, 20, 1,
                                    help="Expected reduction in food waste")
        time_saved = st.slider("Processing Time Saved (%)", 
                              10, 50, 30, 1,
                              help="Reduction in manual processing time")
    
    # Calculate ROI
    st.markdown("---")
    st.markdown("### 📊 ROI Calculation")
    
    annual_transactions = 58016 / 3  # Average per year
    
    # Annual savings
    efficiency_savings = annual_transactions * cost_per_transaction * (efficiency_gain / 100)
    waste_savings = 284173812 / 3 * 0.10 * (waste_reduction / 100)  # $0.10 per unit saved
    time_savings = annual_transactions * 2 * (time_saved / 100)  # $2 per transaction time saved
    
    total_annual_savings = efficiency_savings + waste_savings + time_savings
    net_year1 = total_annual_savings - implementation_cost - annual_maintenance
    net_year2 = total_annual_savings - annual_maintenance
    net_year3 = total_annual_savings - annual_maintenance
    
    cumulative_roi = net_year1 + net_year2 + net_year3
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Annual Savings", f"${total_annual_savings:,.0f}")
    with col2:
        st.metric("3-Year ROI", f"${cumulative_roi:,.0f}", 
                 delta=f"{(cumulative_roi/implementation_cost)*100:.1f}% return")
    with col3:
        payback_months = (implementation_cost / (total_annual_savings - annual_maintenance)) * 12
        st.metric("Payback Period", f"{payback_months:.1f} months")
    with col4:
        roi_percentage = (cumulative_roi / implementation_cost) * 100
        st.metric("ROI Percentage", f"{roi_percentage:.1f}%")
    
    # Detailed breakdown
    st.markdown("---")
    st.markdown("### 💵 Savings Breakdown")
    
    savings_df = pd.DataFrame({
        'Category': ['Efficiency Improvement', 'Waste Reduction', 'Time Savings'],
        'Annual Savings': [f'${efficiency_savings:,.0f}', 
                          f'${waste_savings:,.0f}', 
                          f'${time_savings:,.0f}'],
        'Percentage': [f'{(efficiency_savings/total_annual_savings)*100:.1f}%',
                      f'{(waste_savings/total_annual_savings)*100:.1f}%',
                      f'{(time_savings/total_annual_savings)*100:.1f}%']
    })
    
    st.dataframe(savings_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("## 🎯 Optimization Opportunities")
    
    # Optimization scenarios
    st.markdown("### 🔍 What-If Scenarios")
    
    scenario = st.selectbox("Select Optimization Scenario:", 
                           ["Conservative", "Moderate", "Aggressive"])
    
    if scenario == "Conservative":
        opt_efficiency = 10
        opt_accuracy = 92
        opt_cost_reduction = 15
    elif scenario == "Moderate":
        opt_efficiency = 20
        opt_accuracy = 96
        opt_cost_reduction = 25
    else:  # Aggressive
        opt_efficiency = 35
        opt_accuracy = 99
        opt_cost_reduction = 40
    
    st.info(f"""
    **{scenario} Scenario Assumptions:**
    - Distribution Efficiency: +{opt_efficiency}%
    - Prediction Accuracy: {opt_accuracy}%
    - Cost Reduction: {opt_cost_reduction}%
    """)
    
    # Impact visualization
    st.markdown("---")
    
    current_values = [85.3, 88.0, 12.5]
    optimized_values = [85.3 + opt_efficiency, opt_accuracy, 12.5 * (1 - opt_cost_reduction/100)]
    
    comparison_df = pd.DataFrame({
        'Metric': ['Distribution Efficiency (%)', 'Prediction Accuracy (%)', 'Cost per Transaction ($)'],
        'Current': current_values,
        'Optimized': optimized_values,
        'Improvement': [opt_efficiency, opt_accuracy - 88, -(12.5 * opt_cost_reduction/100)]
    })
    
    st.dataframe(comparison_df.style.format({
        'Current': '{:.2f}',
        'Optimized': '{:.2f}',
        'Improvement': '{:+.2f}'
    }), use_container_width=True)
    
    # Visual comparison
    st.markdown("---")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current',
        x=comparison_df['Metric'],
        y=comparison_df['Current'],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Optimized',
        x=comparison_df['Metric'],
        y=comparison_df['Optimized'],
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title=f'{scenario} Scenario: Current vs Optimized',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Optimization recommendations
    st.markdown("---")
    st.markdown("### 💡 Key Optimization Areas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Quick Wins (0-3 months):**
        - Deploy automated prediction system
        - Implement real-time monitoring
        - Standardize data collection
        - Train staff on new tools
        """)
    
    with col2:
        st.info("""
        **Long-term Goals (3-12 months):**
        - Full automation of routine decisions
        - Advanced forecasting models
        - Integration with supply chain
        - Continuous model improvement
        """)

with tab4:
    st.markdown("## 📈 5-Year Projections")
    
    # 5-year projection
    years = list(range(1, 6))
    
    # Base growth assumptions
    growth_rate = st.slider("Annual Distribution Growth (%)", 0, 10, 3, 1)
    efficiency_improvement = st.slider("Annual Efficiency Improvement (%)", 1, 10, 5, 1)
    
    st.markdown("---")
    
    # Calculate projections
    base_distribution = 284173812
    base_efficiency = 85.3
    
    distributions = []
    efficiencies = []
    savings = []
    cumulative_savings = 0
    
    for year in years:
        dist = base_distribution * ((1 + growth_rate/100) ** year)
        eff = min(99, base_efficiency + (efficiency_improvement * year))
        saving = dist * 0.10 * (eff - base_efficiency) / 100
        cumulative_savings += saving
        
        distributions.append(dist / 1e6)  # Convert to millions
        efficiencies.append(eff)
        savings.append(saving / 1e6)  # Convert to millions
    
    projection_df = pd.DataFrame({
        'Year': years,
        'Distribution (M units)': distributions,
        'Efficiency (%)': efficiencies,
        'Annual Savings ($M)': savings
    })
    
    # Projection charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(projection_df, x='Year', y='Distribution (M units)',
                     title='5-Year Distribution Projection',
                     markers=True)
        fig.update_traces(line_color='steelblue', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(projection_df, x='Year', y='Efficiency (%)',
                     title='5-Year Efficiency Improvement',
                     markers=True)
        fig.update_traces(line_color='green', line_width=3)
        fig.add_hline(y=95, line_dash="dash", line_color="red",
                     annotation_text="Target: 95%")
        st.plotly_chart(fig, use_container_width=True)
    
    # Savings projection
    st.markdown("---")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=projection_df['Year'],
        y=projection_df['Annual Savings ($M)'],
        name='Annual Savings',
        marker_color='lightblue'
    ))
    
    cumulative = projection_df['Annual Savings ($M)'].cumsum()
    fig.add_trace(go.Scatter(
        x=projection_df['Year'],
        y=cumulative,
        name='Cumulative Savings',
        mode='lines+markers',
        line=dict(color='darkblue', width=3),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='5-Year Savings Projection',
        yaxis=dict(title='Annual Savings ($M)'),
        yaxis2=dict(title='Cumulative Savings ($M)', overlaying='y', side='right'),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    st.markdown("---")
    st.markdown("### 📊 5-Year Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Distribution", f"{distributions[-1]:.1f}M units", 
                 delta=f"+{((distributions[-1]/distributions[0])-1)*100:.1f}%")
    with col2:
        st.metric("Final Efficiency", f"{efficiencies[-1]:.1f}%",
                 delta=f"+{efficiencies[-1] - efficiencies[0]:.1f}%")
    with col3:
        st.metric("Cumulative Savings", f"${cumulative_savings/1e6:.2f}M")
    with col4:
        st.metric("Year 5 Annual Savings", f"${savings[-1]:.2f}M")

# Footer
st.markdown("---")
st.success("""
🎯 **Business Impact Summary:**
- Significant cost savings through automation
- Improved distribution efficiency
- Reduced food waste
- Data-driven decision making
- Scalable solution for future growth
""")
