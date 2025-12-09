"""
CausalPilot: The Treatment Effect Simulator
Streamlit App for Interactive HTE Analysis and Targeting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_generator import MarketingDataGenerator
from causal_pipeline import CausalInferencePipeline
from visualization import HTEVisualizer
from targeting_policy import TargetingPolicy

# Page config
st.set_page_config(
    page_title="CausalPilot: HTE Simulator",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 CausalPilot: The Treatment Effect Simulator")
st.markdown("""
**Engineered a Causal Inference Engine using DoWhy/EconML to model and simulate 
the Heterogeneous Treatment Effect (HTE) of marketing campaigns, enabling optimal 
targeting strategies that boost simulated ROI.**
""")

# Sidebar
st.sidebar.header("Configuration")
n_samples = st.sidebar.slider("Number of Samples", 1000, 10000, 5000, 500)
hte_method = st.sidebar.selectbox(
    "HTE Estimation Method",
    ["dml", "forest", "tlearner", "slearner", "xlearner"],
    index=0,
    help="Double Machine Learning (DML) or Causal Forest for HTE estimation"
)
treatment_cost = st.sidebar.number_input("Treatment Cost", 0.1, 10.0, 1.0, 0.1)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'hte_predictions' not in st.session_state:
    st.session_state.hte_predictions = None

# Generate Data
if st.sidebar.button("Generate Data", type="primary"):
    with st.spinner("Generating synthetic marketing data..."):
        generator = MarketingDataGenerator(n_samples=n_samples)
        data = generator.generate()
        st.session_state.data = data
        st.success(f"Generated {len(data)} samples!")

# Run Causal Inference
if st.sidebar.button("Run Causal Inference"):
    if st.session_state.data is None:
        st.error("Please generate data first!")
    else:
        try:
            with st.spinner("Running causal inference pipeline..."):
                # Initialize pipeline
                pipeline = CausalInferencePipeline(
                    st.session_state.data,
                    treatment='treatment',
                    outcome='outcome'
                )
                
                # Identify causal effect
                confounders = ['age', 'income', 'purchase_history', 'engagement_score']
                identified_estimand = pipeline.identify_causal_effect(confounders=confounders)
                st.session_state.identified_estimand = identified_estimand
                
                # Estimate HTE
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Fitting HTE estimator...")
                progress_bar.progress(30)
                
                hte_predictions, X_test = pipeline.estimate_hte(method=hte_method)
                
                progress_bar.progress(80)
                status_text.text("Finalizing results...")
                
                st.session_state.pipeline = pipeline
                st.session_state.hte_predictions = pipeline.hte_predictions_full
                st.session_state.X_test = X_test
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                st.success("Causal inference complete!")
                st.info(f"Average Treatment Effect (ATE): {pipeline.ate:.4f}")
        except Exception as e:
            st.error(f"Error during causal inference: {str(e)}")
            st.exception(e)

# Main content
if st.session_state.data is not None:
    data = st.session_state.data
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🎯 HTE Analysis", 
        "📊 Targeting Segments",
        "💰 ROI Analysis",
        "🔍 Counterfactuals"
    ])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            st.metric("Treatment Rate", f"{data['treatment'].mean()*100:.1f}%")
        with col3:
            st.metric("Conversion Rate", f"{data['outcome'].mean()*100:.1f}%")
        with col4:
            if st.session_state.hte_predictions is not None:
                st.metric("Avg HTE", f"{np.mean(st.session_state.hte_predictions):.4f}")
        
        st.subheader("Sample Data")
        st.dataframe(data.head(100), use_container_width=True)
        
        st.subheader("Data Statistics")
        st.dataframe(data.describe(), use_container_width=True)
    
    with tab2:
        st.header("Heterogeneous Treatment Effect Analysis")
        
        if st.session_state.hte_predictions is None:
            st.warning("Please run causal inference first!")
        else:
            hte_predictions = st.session_state.hte_predictions
            pipeline = st.session_state.pipeline
            
            # Create visualizer
            visualizer = HTEVisualizer(data, hte_predictions)
            
            # HTE Distribution
            st.subheader("HTE Distribution")
            true_hte = data['true_hte'].values if 'true_hte' in data.columns else None
            fig_dist = visualizer.plot_hte_distribution(true_hte=true_hte)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean HTE", f"{np.mean(hte_predictions):.4f}")
            with col2:
                st.metric("Median HTE", f"{np.median(hte_predictions):.4f}")
            with col3:
                st.metric("Min HTE", f"{np.min(hte_predictions):.4f}")
            with col4:
                st.metric("Max HTE", f"{np.max(hte_predictions):.4f}")
            
            # HTE by Feature
            st.subheader("HTE by Feature")
            feature_col = st.selectbox(
                "Select Feature",
                ['age', 'income', 'purchase_history', 'engagement_score'],
                key='feature_select'
            )
            fig_feature = visualizer.plot_hte_by_feature(feature_col)
            st.plotly_chart(fig_feature, use_container_width=True)
            
            # Uplift Curve
            st.subheader("Uplift Curve")
            fig_uplift = visualizer.plot_uplift_curve()
            st.plotly_chart(fig_uplift, use_container_width=True)
    
    with tab3:
        st.header("Targeting Segmentation")
        
        if st.session_state.hte_predictions is None:
            st.warning("Please run causal inference first!")
        else:
            n_segments = st.slider("Number of Segments", 3, 8, 4)
            segments = st.session_state.pipeline.get_targeting_segments(n_segments=n_segments)
            
            # Segment visualization
            visualizer = HTEVisualizer(data, st.session_state.hte_predictions)
            fig_segments = visualizer.plot_targeting_segments(segments)
            st.plotly_chart(fig_segments, use_container_width=True)
            
            # Segment table
            st.subheader("Segment Details")
            segment_df = pd.DataFrame(segments)
            st.dataframe(segment_df, use_container_width=True)
    
    with tab4:
        st.header("ROI Analysis & Optimal Targeting")
        
        if st.session_state.hte_predictions is None:
            st.warning("Please run causal inference first!")
        else:
            policy = TargetingPolicy(data, st.session_state.hte_predictions, 
                                    treatment_cost=treatment_cost)
            
            # Optimal threshold
            st.subheader("Optimal Targeting Threshold")
            optimal = policy.find_optimal_threshold()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Optimal Threshold", f"{optimal['optimal_threshold']:.4f}")
            with col2:
                st.metric("Optimal ROI", f"{optimal['optimal_roi']*100:.2f}%")
            with col3:
                metrics = optimal['metrics']
                st.metric("% Targeted", f"{metrics['percentage_targeted']:.1f}%")
            
            # ROI by threshold
            st.subheader("ROI by Targeting Threshold")
            roi_df = policy.calculate_roi()
            
            fig_roi = go.Figure()
            fig_roi.add_trace(go.Scatter(
                x=roi_df['percentage_targeted'],
                y=roi_df['roi'] * 100,
                mode='lines+markers',
                name='ROI',
                line=dict(color='green', width=2)
            ))
            fig_roi.update_layout(
                title='ROI vs Percentage Targeted',
                xaxis_title='Percentage Targeted (%)',
                yaxis_title='ROI (%)',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_roi, use_container_width=True)
            
            # Comparison with random targeting
            st.subheader("Optimal vs Random Targeting")
            target_pct = st.slider("Target Percentage", 10, 90, 50, 5)
            comparison = policy.compare_with_random_targeting(target_percentage=target_pct)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Optimal ROI",
                    f"{comparison['optimal']['roi']*100:.2f}%",
                    delta=f"{comparison['optimal']['net_benefit']:.2f} net benefit"
                )
            with col2:
                st.metric(
                    "Random ROI",
                    f"{comparison['random']['roi']*100:.2f}%",
                    delta=f"{comparison['random']['net_benefit']:.2f} net benefit"
                )
            
            st.metric(
                "Improvement",
                f"{comparison['improvement_percentage']:.1f}%",
                delta=f"+{comparison['absolute_improvement']:.4f} additional effect"
            )
            
            # ROI table
            st.subheader("Detailed ROI Analysis")
            st.dataframe(roi_df, use_container_width=True)
    
    with tab5:
        st.header("Counterfactual Simulation")
        
        if st.session_state.hte_predictions is None:
            st.warning("Please run causal inference first!")
        else:
            pipeline = st.session_state.pipeline
            
            st.subheader("Simulate Counterfactual Outcomes")
            treatment_scenario = st.selectbox(
                "Treatment Scenario",
                ["Treat All", "Treat None", "Optimal Targeting"],
                key='counterfactual_scenario'
            )
            
            if st.button("Simulate Counterfactual"):
                with st.spinner("Simulating counterfactuals..."):
                    # Prepare feature matrix
                    X = data.drop(['treatment', 'outcome', 'true_hte'], axis=1, errors='ignore')
                    X = pd.get_dummies(X, drop_first=True).values
                    
                    if treatment_scenario == "Treat All":
                        counterfactual_outcomes = pipeline.simulate_counterfactuals(X, treatment_value=1)
                        expected_conversions = np.mean(counterfactual_outcomes) * len(data)
                    elif treatment_scenario == "Treat None":
                        counterfactual_outcomes = pipeline.simulate_counterfactuals(X, treatment_value=0)
                        expected_conversions = np.mean(counterfactual_outcomes) * len(data)
                    else:
                        # Optimal targeting
                        policy = TargetingPolicy(data, st.session_state.hte_predictions, 
                                                treatment_cost=treatment_cost)
                        optimal = policy.find_optimal_threshold()
                        targeted = policy.get_targeting_list(
                            threshold=optimal['optimal_threshold']
                        )
                        expected_conversions = len(targeted) * (np.mean(targeted['predicted_hte']) + 
                                                               data[data['treatment']==0]['outcome'].mean())
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Scenario", treatment_scenario)
                    with col2:
                        st.metric("Expected Conversions", f"{expected_conversions:.0f}")
                    with col3:
                        baseline_conversions = len(data) * data['outcome'].mean()
                        improvement = ((expected_conversions - baseline_conversions) / baseline_conversions) * 100
                        st.metric("vs Baseline", f"{improvement:+.1f}%")
else:
    st.info("👈 Please generate data using the sidebar to get started!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
**CausalPilot** demonstrates:
- Causal inference with DoWhy
- HTE estimation with EconML
- Counterfactual simulation
- Optimal targeting strategies
""")

