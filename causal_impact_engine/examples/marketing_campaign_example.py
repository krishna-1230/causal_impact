"""
Example script for analyzing marketing campaign data.
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Add the parent directory to the path to import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from causal_impact_engine.data.sample_data import SampleData
from causal_impact_engine.models.model_factory import ModelFactory
from causal_impact_engine.utils.metrics import CausalImpactMetrics
from causal_impact_engine.utils.reporting import CausalImpactReporter
from causal_impact_engine.utils.visualization import CausalImpactVisualizer


def main():
    """Run the marketing campaign example."""
    print("Causal Impact Engine: Marketing Campaign Example")
    print("=" * 50)
    
    # Generate sample marketing campaign data
    print("\nGenerating sample marketing campaign data...")
    data = SampleData.get_marketing_campaign_data(random_seed=42)
    
    # Display the data
    print("\nSample of the data:")
    print(data.head())
    
    # Plot the data
    print("\nPlotting the time series data...")
    fig = CausalImpactVisualizer.plot_time_series(
        data=data,
        date_col="date",
        target_col="sales",
        intervention_date="2022-04-01",
        title="Marketing Campaign Data"
    )
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "marketing_campaign_data.png"))
    plt.close(fig)
    
    # Define pre and post-intervention periods
    pre_period = ["2022-01-01", "2022-03-31"]
    post_period = ["2022-04-01", "2022-06-30"]
    
    # Run causal impact analysis using CausalImpact
    print("\nRunning causal impact analysis using CausalImpact...")
    model = ModelFactory.create_model(
        model_type="causalimpact",
        data=data,
        pre_period=pre_period,
        post_period=post_period,
        target_col="sales",
        date_col="date",
        covariates=["web_traffic", "ad_spend"]
    )
    
    # Run inference
    model.run_inference()
    
    # Get summary
    summary = model.get_summary()
    print("\nCausal Impact Summary:")
    print(f"Relative Effect: {summary['relative_effect'] * 100:.2f}%")
    print(f"Absolute Effect: {summary['average_effect']:.2f}")
    print(f"Cumulative Effect: {summary['cumulative_effect']:.2f}")
    print(f"p-value: {summary['p_value']:.4f}")
    print(f"Statistically Significant: {summary['p_value'] < 0.05}")
    
    # Plot results
    print("\nPlotting causal impact results...")
    fig = model.plot_results()
    fig.savefig(os.path.join(output_dir, "causal_impact_results.png"))
    plt.close(fig)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    predictions = model.predict()
    metrics = CausalImpactMetrics.calculate_all_metrics(
        data=predictions,
        target_col="sales",
        prediction_col="prediction",
        intervention_col="campaign"
    )
    
    # Generate reports
    print("\nGenerating reports...")
    summary_report = CausalImpactReporter.generate_summary_report(
        model_results=summary,
        detailed=True
    )
    
    technical_report = CausalImpactReporter.generate_technical_report(
        data=predictions,
        target_col="sales",
        prediction_col="prediction",
        intervention_col="campaign"
    )
    
    executive_summary = CausalImpactReporter.generate_executive_summary(
        model_results=summary,
        intervention_name="marketing campaign",
        target_name="sales"
    )
    
    # Save reports
    with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
        f.write(summary_report)
        
    with open(os.path.join(output_dir, "technical_report.txt"), "w") as f:
        f.write(technical_report)
        
    with open(os.path.join(output_dir, "executive_summary.txt"), "w") as f:
        f.write(executive_summary)
    
    # Generate HTML report
    html_report = CausalImpactReporter.generate_html_report(
        model_results=summary,
        metrics=metrics,
        intervention_name="marketing campaign",
        target_name="sales"
    )
    
    with open(os.path.join(output_dir, "report.html"), "w") as f:
        f.write(html_report)
    
    # Create interactive plot
    print("\nCreating interactive visualization...")
    interactive_plot = CausalImpactVisualizer.create_interactive_plot(
        data=predictions,
        date_col="date",
        target_col="sales",
        counterfactual_col="prediction",
        lower_col="prediction_lower",
        upper_col="prediction_upper",
        effect_col="effect",
        intervention_date="2022-04-01",
        title="Marketing Campaign Causal Impact Analysis"
    )
    
    # Save interactive plot as HTML
    interactive_plot.write_html(os.path.join(output_dir, "interactive_plot.html"))
    
    print("\nExample completed successfully!")
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main() 