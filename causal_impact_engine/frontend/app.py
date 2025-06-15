"""
Streamlit web application for the causal impact engine.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add the parent directory to the path to import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from causal_impact_engine.data.data_generator import DataGenerator
from causal_impact_engine.data.data_loader import DataLoader
from causal_impact_engine.data.sample_data import SampleData
from causal_impact_engine.models.model_factory import ModelFactory
from causal_impact_engine.utils.metrics import CausalImpactMetrics
from causal_impact_engine.utils.reporting import CausalImpactReporter
from causal_impact_engine.utils.visualization import CausalImpactVisualizer


# Set page configuration
st.set_page_config(
    page_title="Causal Impact Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Run the Streamlit app."""
    # Header
    st.title("🧠 Causal Impact Engine")
    st.subheader("Marketing Attribution Simulator")
    
    st.markdown("""
    This application helps you analyze the causal impact of marketing campaigns or other interventions
    on your business metrics. Using Bayesian structural time series models, it can determine if your
    campaign actually caused an increase in sales or other metrics.
    """)
    
    # Sidebar
    st.sidebar.title("Options")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Sample Data", "Upload CSV", "Generate Synthetic Data"]
    )
    
    # Initialize data
    data = None
    
    # Handle data source selection
    if data_source == "Sample Data":
        sample_dataset = st.sidebar.selectbox(
            "Select Sample Dataset",
            ["Marketing Campaign", "E-commerce Website Redesign"]
        )
        
        if sample_dataset == "Marketing Campaign":
            data = SampleData.get_marketing_campaign_data()
            date_col = "date"
            target_col = "sales"
            intervention_col = "campaign"
            intervention_date = "2022-04-01"
            covariates = ["web_traffic", "ad_spend"]
            
        elif sample_dataset == "E-commerce Website Redesign":
            data = SampleData.get_ecommerce_data()
            date_col = "date"
            target_col = "revenue"
            intervention_col = "redesign"
            intervention_date = "2022-04-01"
            covariates = ["visitors", "conversion_rate", "aov"]
    
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.sidebar.success("File uploaded successfully!")
                
                # Get column names
                columns = data.columns.tolist()
                
                # Select columns
                date_col = st.sidebar.selectbox("Select Date Column", columns)
                target_col = st.sidebar.selectbox("Select Target Column", columns)
                intervention_col = st.sidebar.selectbox("Select Intervention Column", columns)
                
                # Convert date column to datetime
                data[date_col] = pd.to_datetime(data[date_col])
                
                # Get intervention date
                intervention_dates = data[data[intervention_col] == 1][date_col].unique()
                if len(intervention_dates) > 0:
                    intervention_date = intervention_dates[0].strftime("%Y-%m-%d")
                else:
                    intervention_date = st.sidebar.date_input(
                        "Intervention Date",
                        datetime.now()
                    ).strftime("%Y-%m-%d")
                
                # Select covariates
                available_covariates = [col for col in columns if col not in [date_col, target_col, intervention_col]]
                covariates = st.sidebar.multiselect("Select Covariates (Optional)", available_covariates)
                
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
                data = None
    
    elif data_source == "Generate Synthetic Data":
        st.sidebar.subheader("Synthetic Data Parameters")
        
        pre_period_days = st.sidebar.slider("Pre-intervention Days", 30, 180, 90)
        post_period_days = st.sidebar.slider("Post-intervention Days", 10, 90, 30)
        effect_size = st.sidebar.slider("Effect Size", -0.5, 0.5, 0.2, 0.01)
        noise_level = st.sidebar.slider("Noise Level", 0.01, 0.5, 0.1, 0.01)
        
        data = DataGenerator.generate_synthetic_data(
            pre_period_days=pre_period_days,
            post_period_days=post_period_days,
            intervention_date="2023-01-01",
            effect_size=effect_size,
            noise_level=noise_level,
            random_seed=42
        )
        
        date_col = "date"
        target_col = "y"
        intervention_col = "intervention"
        intervention_date = "2023-01-01"
        covariates = ["x1", "x2"]
    
    # If we have data, proceed with analysis
    if data is not None:
        # Display data
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        # Plot the data
        st.subheader("Time Series Visualization")
        
        # Create two columns for the plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig = CausalImpactVisualizer.plot_time_series(
                data=data,
                date_col=date_col,
                target_col=target_col,
                intervention_date=intervention_date,
                title=f"{target_col} Over Time"
            )
            st.pyplot(fig)
        
        # Model parameters
        st.sidebar.subheader("Model Parameters")
        
        model_type = st.sidebar.selectbox(
            "Select Model Type",
            ["causalimpact", "pymc"]
        )
        
        # Define pre and post periods
        intervention_dt = pd.to_datetime(intervention_date)
        min_date = data[date_col].min()
        max_date = data[date_col].max()
        
        pre_end = intervention_dt - pd.Timedelta(days=1)
        pre_period = [min_date.strftime("%Y-%m-%d"), pre_end.strftime("%Y-%m-%d")]
        post_period = [intervention_date, max_date.strftime("%Y-%m-%d")]
        
        # Run button
        run_analysis = st.sidebar.button("Run Causal Impact Analysis")
        
        if run_analysis:
            with st.spinner("Running causal impact analysis..."):
                # Create model
                model = ModelFactory.create_model(
                    model_type=model_type,
                    data=data,
                    pre_period=pre_period,
                    post_period=post_period,
                    target_col=target_col,
                    date_col=date_col,
                    covariates=covariates
                )
                
                # Run inference
                model.run_inference()
                
                # Get summary
                summary = model.get_summary()
                
                # Get predictions
                predictions = model.predict()
                
                # Calculate metrics
                metrics = CausalImpactMetrics.calculate_all_metrics(
                    data=predictions,
                    target_col=target_col,
                    prediction_col="prediction",
                    intervention_col="intervention"
                )
                
                # Display results
                st.subheader("Causal Impact Analysis Results")
                
                # Create tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Visualization", "Metrics", "Report"])
                
                with tab1:
                    # Display summary
                    st.markdown("### Impact Summary")
                    
                    # Create two columns for the summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="Relative Effect",
                            value=f"{summary['relative_effect'] * 100:.2f}%",
                            delta=f"{summary['relative_effect'] * 100:.2f}%"
                        )
                        
                        st.metric(
                            label="Absolute Effect",
                            value=f"{summary['average_effect']:.2f}",
                            delta=f"{summary['average_effect']:.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            label="Cumulative Effect",
                            value=f"{summary['cumulative_effect']:.2f}",
                            delta=f"{summary['cumulative_effect']:.2f}"
                        )
                        
                        st.metric(
                            label="p-value",
                            value=f"{summary['p_value']:.4f}",
                            delta="Significant" if summary['p_value'] < 0.05 else "Not Significant"
                        )
                    
                    # Display executive summary
                    st.markdown("### Executive Summary")
                    executive_summary = CausalImpactReporter.generate_executive_summary(
                        model_results=summary,
                        intervention_name="intervention",
                        target_name=target_col
                    )
                    st.text_area("", executive_summary, height=300)
                
                with tab2:
                    # Display visualizations
                    st.markdown("### Causal Impact Visualization")
                    
                    # Create interactive plot
                    interactive_plot = CausalImpactVisualizer.create_interactive_plot(
                        data=predictions,
                        date_col=date_col,
                        target_col=target_col,
                        counterfactual_col="prediction",
                        lower_col="prediction_lower",
                        upper_col="prediction_upper",
                        effect_col="effect",
                        intervention_date=intervention_date,
                        title="Causal Impact Analysis"
                    )
                    
                    st.plotly_chart(interactive_plot, use_container_width=True)
                    
                    # Add matplotlib plot
                    if model_type == "pymc":
                        st.markdown("### Posterior Distribution")
                        fig = model.plot_posterior()
                        st.pyplot(fig)
                
                with tab3:
                    # Display metrics
                    st.markdown("### Model Fit Metrics")
                    
                    # Create two columns for the metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(label="RMSE (Pre-Period)", value=f"{metrics['pre_rmse']:.4f}")
                        st.metric(label="MAE (Pre-Period)", value=f"{metrics['pre_mae']:.4f}")
                    
                    with col2:
                        st.metric(label="R-squared (Pre-Period)", value=f"{metrics['pre_r_squared']:.4f}")
                        if "pre_mape" in metrics:
                            st.metric(label="MAPE (Pre-Period)", value=f"{metrics['pre_mape']:.2f}%")
                    
                    st.markdown("### Effect Size Metrics")
                    
                    # Create two columns for the effect metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="Absolute Effect",
                            value=f"{metrics['absolute_effect']:.4f}"
                        )
                        
                        st.metric(
                            label="Relative Effect",
                            value=f"{metrics['relative_effect'] * 100:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            label="Cumulative Effect",
                            value=f"{metrics['cumulative_effect']:.4f}"
                        )
                        
                        st.metric(
                            label="Cohen's d",
                            value=f"{metrics['cohens_d']:.4f}"
                        )
                    
                    st.markdown("### Statistical Significance")
                    
                    # Create two columns for the significance metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="t-statistic",
                            value=f"{metrics['t_statistic']:.4f}"
                        )
                        
                        st.metric(
                            label="p-value",
                            value=f"{metrics['p_value']:.4f}"
                        )
                    
                    with col2:
                        st.metric(
                            label="z-score",
                            value=f"{metrics['z_score']:.4f}"
                        )
                        
                        st.metric(
                            label="Statistically Significant",
                            value="Yes" if metrics['is_significant'] else "No"
                        )
                
                with tab4:
                    # Display technical report
                    st.markdown("### Technical Report")
                    technical_report = CausalImpactReporter.generate_technical_report(
                        data=predictions,
                        target_col=target_col,
                        prediction_col="prediction",
                        intervention_col="intervention"
                    )
                    st.text_area("", technical_report, height=400)
                    
                    # Add download buttons
                    st.markdown("### Download Reports")
                    
                    # Generate HTML report
                    html_report = CausalImpactReporter.generate_html_report(
                        model_results=summary,
                        metrics=metrics,
                        intervention_name="intervention",
                        target_name=target_col
                    )
                    
                    # Create columns for download buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label="Download HTML Report",
                            data=html_report,
                            file_name="causal_impact_report.html",
                            mime="text/html"
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download Technical Report",
                            data=technical_report,
                            file_name="technical_report.txt",
                            mime="text/plain"
                        )
                    
                    with col3:
                        st.download_button(
                            label="Download Executive Summary",
                            data=executive_summary,
                            file_name="executive_summary.txt",
                            mime="text/plain"
                        )
    
    # Footer
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Causal Impact Engine** is a tool for analyzing the causal impact of interventions using Bayesian structural time series models.
    
    It helps answer the question: "Did my marketing campaign cause an increase in sales?" by comparing the observed data after the intervention
    with a counterfactual prediction of what would have happened without the intervention.
    
    The methodology is based on [Google's CausalImpact package](https://google.github.io/CausalImpact/) and Bayesian structural time series models.
    """)


if __name__ == "__main__":
    main() 