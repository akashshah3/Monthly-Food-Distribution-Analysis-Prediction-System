import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import get_recommendations

st.set_page_config(page_title="Recommendations", page_icon="🎯", layout="wide")

st.title("🎯 Strategic Recommendations")
st.markdown("### Data-driven action plan for optimization")

# Get recommendations
recommendations = get_recommendations()

# Tab selection
tab1, tab2, tab3 = st.tabs(["📋 All Recommendations", "🎯 Priority Matrix", "📅 Implementation Plan"])

with tab1:
    st.markdown("## 📋 Actionable Recommendations")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category_filter = st.multiselect(
            "Filter by Category:",
            options=recommendations['Category'].unique().tolist(),
            default=recommendations['Category'].unique().tolist()
        )
    
    with col2:
        priority_filter = st.multiselect(
            "Filter by Priority:",
            options=recommendations['Priority'].unique().tolist(),
            default=recommendations['Priority'].unique().tolist()
        )
    
    with col3:
        impact_filter = st.multiselect(
            "Filter by Impact:",
            options=recommendations['Impact'].unique().tolist(),
            default=recommendations['Impact'].unique().tolist()
        )
    
    # Apply filters
    filtered_df = recommendations[
        (recommendations['Category'].isin(category_filter)) &
        (recommendations['Priority'].isin(priority_filter)) &
        (recommendations['Impact'].isin(impact_filter))
    ]
    
    st.info(f"Showing {len(filtered_df)} of {len(recommendations)} recommendations")
    
    # Display recommendations as cards
    st.markdown("---")
    
    for idx, row in filtered_df.iterrows():
        # Priority color mapping
        priority_colors = {
            'Critical': '🔴',
            'High': '🟠',
            'Medium': '🟡',
            'Low': '🟢'
        }
        
        with st.expander(f"{priority_colors.get(row['Priority'], '⚪')} **{row['Recommendation']}** ({row['Category']})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Category:** {row['Category']}")
                st.markdown(f"**Priority:** {row['Priority']}")
                st.markdown(f"**Impact:** {row['Impact']}")
                st.markdown(f"**Effort:** {row['Effort']}")
                st.markdown(f"**Timeline:** {row['Timeline']}")
            
            with col2:
                # Impact score visualization
                impact_scores = {'High': 3, 'Medium': 2, 'Low': 1}
                effort_scores = {'Low': 1, 'Medium': 2, 'High': 3}
                
                impact_val = impact_scores.get(row['Impact'], 2)
                effort_val = effort_scores.get(row['Effort'], 2)
                
                st.metric("Impact Score", f"{impact_val}/3")
                st.metric("Effort Score", f"{effort_val}/3")
                
                # ROI indicator (Impact/Effort)
                roi_score = impact_val / effort_val
                st.metric("ROI Indicator", f"{roi_score:.2f}")
    
    # Summary statistics
    st.markdown("---")
    st.markdown("### 📊 Recommendations Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        critical_count = len(filtered_df[filtered_df['Priority'] == 'Critical'])
        st.metric("Critical Priority", critical_count)
    
    with col2:
        high_impact_count = len(filtered_df[filtered_df['Impact'] == 'High'])
        st.metric("High Impact", high_impact_count)
    
    with col3:
        low_effort_count = len(filtered_df[filtered_df['Effort'] == 'Low'])
        st.metric("Low Effort", low_effort_count)
    
    with col4:
        quick_wins = len(filtered_df[(filtered_df['Impact'] == 'High') & 
                                     (filtered_df['Effort'].isin(['Low', 'Medium']))])
        st.metric("Quick Wins", quick_wins)

with tab2:
    st.markdown("## 🎯 Impact vs Effort Priority Matrix")
    
    st.info("""
    **How to read this matrix:**
    - **Top Right (High Impact, Low Effort):** Quick wins - prioritize these!
    - **Top Left (High Impact, High Effort):** Major projects - plan carefully
    - **Bottom Right (Low Impact, Low Effort):** Fill-ins - do when time permits
    - **Bottom Left (Low Impact, High Effort):** Consider deprioritizing
    """)
    
    # Prepare data for scatter plot
    impact_mapping = {'High': 3, 'Medium': 2, 'Low': 1}
    effort_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    
    matrix_df = recommendations.copy()
    matrix_df['Impact_Score'] = matrix_df['Impact'].map(impact_mapping)
    matrix_df['Effort_Score'] = matrix_df['Effort'].map(effort_mapping)
    matrix_df['ROI'] = matrix_df['Impact_Score'] / matrix_df['Effort_Score']
    
    # Add quadrant labels
    def get_quadrant(row):
        if row['Impact_Score'] >= 2.5 and row['Effort_Score'] <= 1.5:
            return 'Quick Wins'
        elif row['Impact_Score'] >= 2.5 and row['Effort_Score'] > 1.5:
            return 'Major Projects'
        elif row['Impact_Score'] < 2.5 and row['Effort_Score'] <= 1.5:
            return 'Fill-ins'
        else:
            return 'Low Priority'
    
    matrix_df['Quadrant'] = matrix_df.apply(get_quadrant, axis=1)
    
    # Create scatter plot
    fig = px.scatter(matrix_df, 
                    x='Effort_Score', 
                    y='Impact_Score',
                    color='Quadrant',
                    size='ROI',
                    hover_data=['Recommendation', 'Category', 'Priority'],
                    text=matrix_df.index + 1,  # Show numbers
                    title='Priority Matrix: Impact vs Effort',
                    labels={'Effort_Score': 'Effort', 'Impact_Score': 'Impact'},
                    color_discrete_map={
                        'Quick Wins': 'green',
                        'Major Projects': 'blue',
                        'Fill-ins': 'orange',
                        'Low Priority': 'gray'
                    })
    
    # Add quadrant lines
    fig.add_hline(y=2.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=1.5, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=1, y=3, text="Quick Wins", showarrow=False, 
                      font=dict(size=14, color="green"), opacity=0.5)
    fig.add_annotation(x=2.5, y=3, text="Major Projects", showarrow=False,
                      font=dict(size=14, color="blue"), opacity=0.5)
    fig.add_annotation(x=1, y=1, text="Fill-ins", showarrow=False,
                      font=dict(size=14, color="orange"), opacity=0.5)
    fig.add_annotation(x=2.5, y=1, text="Low Priority", showarrow=False,
                      font=dict(size=14, color="gray"), opacity=0.5)
    
    fig.update_traces(textposition='top center', textfont_size=10)
    fig.update_layout(height=600)
    fig.update_xaxes(range=[0.5, 3.5], tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High'])
    fig.update_yaxes(range=[0.5, 3.5], tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High'])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Quadrant breakdown
    st.markdown("---")
    st.markdown("### 📊 Quadrant Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        quick_wins = matrix_df[matrix_df['Quadrant'] == 'Quick Wins']
        st.success(f"**Quick Wins: {len(quick_wins)}**")
        for idx, row in quick_wins.iterrows():
            st.write(f"{idx+1}. {row['Recommendation'][:40]}...")
    
    with col2:
        major_projects = matrix_df[matrix_df['Quadrant'] == 'Major Projects']
        st.info(f"**Major Projects: {len(major_projects)}**")
        for idx, row in major_projects.iterrows():
            st.write(f"{idx+1}. {row['Recommendation'][:40]}...")
    
    with col3:
        fill_ins = matrix_df[matrix_df['Quadrant'] == 'Fill-ins']
        st.warning(f"**Fill-ins: {len(fill_ins)}**")
        for idx, row in fill_ins.iterrows():
            st.write(f"{idx+1}. {row['Recommendation'][:40]}...")
    
    with col4:
        low_priority = matrix_df[matrix_df['Quadrant'] == 'Low Priority']
        st.error(f"**Low Priority: {len(low_priority)}**")
        for idx, row in low_priority.iterrows():
            st.write(f"{idx+1}. {row['Recommendation'][:40]}...")

with tab3:
    st.markdown("## 📅 Implementation Timeline")
    
    # Timeline visualization
    st.info("**Suggested Implementation Sequence by Priority**")
    
    # Sort by priority
    priority_order = {'Critical': 1, 'High': 2, 'Medium': 3, 'Low': 4}
    timeline_df = recommendations.copy()
    timeline_df['Priority_Order'] = timeline_df['Priority'].map(priority_order)
    timeline_df = timeline_df.sort_values('Priority_Order')
    
    # Create Gantt-style chart
    timeline_data = []
    start_month = 0
    
    for idx, row in timeline_df.iterrows():
        # Parse timeline
        timeline_str = row['Timeline']
        if 'month' in timeline_str.lower():
            duration = int(timeline_str.split('-')[0])
        elif 'week' in timeline_str.lower():
            duration = 0.5
        else:
            duration = 3  # Default
        
        timeline_data.append({
            'Recommendation': row['Recommendation'][:50] + '...' if len(row['Recommendation']) > 50 else row['Recommendation'],
            'Start': start_month,
            'End': start_month + duration,
            'Priority': row['Priority'],
            'Category': row['Category']
        })
        
        start_month += duration
    
    timeline_df_plot = pd.DataFrame(timeline_data)
    
    fig = px.timeline(timeline_df_plot,
                     x_start='Start',
                     x_end='End',
                     y='Recommendation',
                     color='Priority',
                     color_discrete_map={
                         'Critical': 'red',
                         'High': 'orange',
                         'Medium': 'yellow',
                         'Low': 'green'
                     },
                     title='Implementation Timeline (Sequential)')
    
    fig.update_yaxes(categoryorder='total ascending')
    fig.update_layout(height=max(400, len(timeline_df_plot) * 30))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Phase-based implementation
    st.markdown("---")
    st.markdown("### 🗓️ Phase-Based Implementation Plan")
    
    # Group by timeline
    phase1 = timeline_df[timeline_df['Timeline'].str.contains('1-2|Immediate', case=False)]
    phase2 = timeline_df[timeline_df['Timeline'].str.contains('3-4', case=False)]
    phase3 = timeline_df[timeline_df['Timeline'].str.contains('6|Ongoing', case=False)]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🚀 Phase 1 (0-2 months)")
        st.success(f"**{len(phase1)} recommendations**")
        for idx, row in phase1.iterrows():
            st.write(f"✓ {row['Recommendation'][:60]}...")
    
    with col2:
        st.markdown("#### 📈 Phase 2 (3-6 months)")
        st.info(f"**{len(phase2)} recommendations**")
        for idx, row in phase2.iterrows():
            st.write(f"⏳ {row['Recommendation'][:60]}...")
    
    with col3:
        st.markdown("#### 🎯 Phase 3 (6+ months)")
        st.warning(f"**{len(phase3)} recommendations**")
        for idx, row in phase3.iterrows():
            st.write(f"📋 {row['Recommendation'][:60]}...")
    
    # Resource allocation
    st.markdown("---")
    st.markdown("### 💼 Resource Allocation Estimate")
    
    resource_df = pd.DataFrame({
        'Phase': ['Phase 1 (0-2m)', 'Phase 2 (3-6m)', 'Phase 3 (6+m)'],
        'Personnel': [3, 5, 2],
        'Budget ($K)': [50, 120, 80],
        'Time (weeks)': [8, 16, 24]
    })
    
    st.dataframe(resource_df, use_container_width=True, hide_index=True)
    
    # Category distribution
    col1, col2 = st.columns(2)
    
    with col1:
        category_counts = recommendations['Category'].value_counts()
        fig = px.pie(values=category_counts.values, 
                    names=category_counts.index,
                    title='Recommendations by Category')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        priority_counts = recommendations['Priority'].value_counts()
        fig = px.bar(x=priority_counts.index, 
                    y=priority_counts.values,
                    title='Recommendations by Priority',
                    color=priority_counts.index,
                    color_discrete_map={
                        'Critical': 'red',
                        'High': 'orange',
                        'Medium': 'yellow',
                        'Low': 'green'
                    })
        st.plotly_chart(fig, use_container_width=True)


# Footer
st.markdown("---")
st.success("""
🎯 **Key Takeaways:**
- Focus on **Quick Wins** for immediate impact
- Plan **Major Projects** with adequate resources
- Implement in **phases** to manage complexity
- Monitor progress with **clear KPIs**
- Maintain **continuous improvement** mindset
""")

st.info("💡 **Pro Tip:** Start with high-priority, low-effort recommendations to build momentum and demonstrate value quickly!")
