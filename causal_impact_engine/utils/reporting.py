"""
Reporting utilities for generating causal impact reports.
"""
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from causal_impact_engine.utils.metrics import CausalImpactMetrics


class CausalImpactReporter:
    """Generate reports for causal impact analysis."""
    
    @staticmethod
    def generate_summary_report(
        model_results: Dict,
        alpha: float = 0.05,
        detailed: bool = False,
    ) -> str:
        """
        Generate a summary report of the causal impact analysis.
        
        Args:
            model_results: Dictionary of model results
            alpha: Significance level (default: 0.05)
            detailed: Whether to include detailed statistics
            
        Returns:
            Formatted report as string
        """
        # Extract key metrics
        rel_effect = model_results.get("relative_effect", 0) * 100
        abs_effect = model_results.get("average_effect", 0)
        cum_effect = model_results.get("cumulative_effect", 0)
        p_value = model_results.get("p_value", 1)
        
        # Determine the direction of the effect
        if rel_effect > 0:
            direction = "positive"
            verb = "increased"
        else:
            direction = "negative"
            verb = "decreased"
            
        # Format confidence interval
        ci_lower = model_results.get("relative_effect_lower", 0) * 100
        ci_upper = model_results.get("relative_effect_upper", 0) * 100
        
        # Generate report
        report = (
            f"Causal Impact Analysis Summary\n"
            f"==============================\n\n"
            f"During the post-intervention period, the response variable {verb} by "
            f"{abs(rel_effect):.1f}% ({ci_lower:.1f}% to {ci_upper:.1f}%).\n\n"
        )
        
        # Add absolute and cumulative effects
        report += (
            f"In absolute terms, the average effect was {abs_effect:.2f} "
            f"with a cumulative impact of {cum_effect:.2f} over the post-intervention period.\n\n"
        )
        
        # Add statistical significance
        report += (
            f"The probability of observing this effect by chance is p = {p_value:.3f}. "
            f"This means that the causal effect can be considered statistically "
        )
        
        if p_value < alpha:
            report += "significant.\n\n"
        else:
            report += "insignificant.\n\n"
        
        # Add interpretation
        if p_value < alpha:
            if rel_effect > 0:
                report += (
                    f"The analysis suggests that the intervention had a positive causal effect "
                    f"on the response variable, increasing it by approximately {abs(rel_effect):.1f}%.\n\n"
                )
            else:
                report += (
                    f"The analysis suggests that the intervention had a negative causal effect "
                    f"on the response variable, decreasing it by approximately {abs(rel_effect):.1f}%.\n\n"
                )
        else:
            report += (
                f"The analysis does not provide sufficient evidence to conclude that "
                f"the intervention had a causal effect on the response variable.\n\n"
            )
        
        # Add detailed statistics if requested
        if detailed:
            report += (
                f"Detailed Statistics\n"
                f"------------------\n"
                f"Average Effect: {abs_effect:.4f}\n"
                f"95% CI: [{model_results.get('average_effect_lower', 0):.4f}, "
                f"{model_results.get('average_effect_upper', 0):.4f}]\n"
                f"Cumulative Effect: {cum_effect:.4f}\n"
                f"Relative Effect: {rel_effect:.2f}%\n"
                f"p-value: {p_value:.4f}\n"
                f"Posterior Probability: {model_results.get('posterior_probability', 0) * 100:.1f}%\n"
            )
        
        return report
    
    @staticmethod
    def generate_technical_report(
        data: pd.DataFrame,
        target_col: str = "y",
        prediction_col: str = "prediction",
        intervention_col: str = "intervention",
    ) -> str:
        """
        Generate a technical report with detailed metrics.
        
        Args:
            data: DataFrame with actual and predicted values
            target_col: Name of the target column
            prediction_col: Name of the prediction column
            intervention_col: Name of the intervention indicator column
            
        Returns:
            Technical report as string
        """
        # Calculate all metrics
        metrics = CausalImpactMetrics.calculate_all_metrics(
            data, target_col, prediction_col, intervention_col
        )
        
        # Generate report
        report = (
            f"Causal Impact Technical Report\n"
            f"============================\n\n"
            f"Model Fit Metrics (Pre-Intervention Period)\n"
            f"------------------------------------------\n"
            f"RMSE: {metrics.get('pre_rmse', 0):.4f}\n"
            f"MAE: {metrics.get('pre_mae', 0):.4f}\n"
            f"R-squared: {metrics.get('pre_r_squared', 0):.4f}\n"
        )
        
        # Add MAPE if available
        if "pre_mape" in metrics:
            report += f"MAPE: {metrics.get('pre_mape', 0):.4f}%\n"
            
        report += (
            f"\nEffect Size Metrics\n"
            f"------------------\n"
            f"Absolute Effect: {metrics.get('absolute_effect', 0):.4f}\n"
            f"Relative Effect: {metrics.get('relative_effect', 0) * 100:.2f}%\n"
            f"Cumulative Effect: {metrics.get('cumulative_effect', 0):.4f}\n"
            f"Cohen's d: {metrics.get('cohens_d', 0):.4f}\n"
            f"\nStatistical Significance\n"
            f"------------------------\n"
            f"t-statistic: {metrics.get('t_statistic', 0):.4f}\n"
            f"p-value: {metrics.get('p_value', 0):.4f}\n"
            f"z-score: {metrics.get('z_score', 0):.4f}\n"
            f"Statistically Significant: {metrics.get('is_significant', False)}\n"
        )
        
        return report
    
    @staticmethod
    def generate_executive_summary(
        model_results: Dict,
        intervention_name: str = "intervention",
        target_name: str = "target variable",
    ) -> str:
        """
        Generate an executive summary of the causal impact analysis.
        
        Args:
            model_results: Dictionary of model results
            intervention_name: Name of the intervention
            target_name: Name of the target variable
            
        Returns:
            Executive summary as string
        """
        # Extract key metrics
        rel_effect = model_results.get("relative_effect", 0) * 100
        cum_effect = model_results.get("cumulative_effect", 0)
        p_value = model_results.get("p_value", 1)
        
        # Determine the direction and significance
        if abs(rel_effect) < 1:
            magnitude = "negligible"
        elif abs(rel_effect) < 5:
            magnitude = "small"
        elif abs(rel_effect) < 10:
            magnitude = "moderate"
        else:
            magnitude = "large"
            
        if rel_effect > 0:
            direction = "positive"
            verb = "increased"
        else:
            direction = "negative"
            verb = "decreased"
            
        significant = p_value < 0.05
        
        # Generate executive summary
        summary = (
            f"Executive Summary: Impact of {intervention_name} on {target_name}\n"
            f"=================================================================\n\n"
        )
        
        if significant:
            summary += (
                f"The {intervention_name} had a statistically significant {direction} impact "
                f"on {target_name}, which {verb} by {abs(rel_effect):.1f}%. "
                f"This represents a {magnitude} effect size.\n\n"
                f"Key findings:\n"
                f"- The {target_name} {verb} by {abs(rel_effect):.1f}% as a result of the {intervention_name}.\n"
                f"- The cumulative impact over the entire post-intervention period was {cum_effect:.1f}.\n"
                f"- There is strong statistical evidence (p = {p_value:.3f}) that this effect was caused by "
                f"the {intervention_name} rather than by random fluctuations.\n\n"
            )
            
            if rel_effect > 0:
                summary += (
                    f"Recommendation: The {intervention_name} appears to be effective and should be "
                    f"considered for continuation or expansion.\n"
                )
            else:
                summary += (
                    f"Recommendation: The {intervention_name} appears to have a negative effect and "
                    f"should be reconsidered or modified.\n"
                )
        else:
            summary += (
                f"The {intervention_name} did not have a statistically significant impact "
                f"on {target_name}. Although we observed a {direction} change of {abs(rel_effect):.1f}%, "
                f"this effect cannot be reliably attributed to the {intervention_name} (p = {p_value:.3f}).\n\n"
                f"Key findings:\n"
                f"- No statistically significant change in {target_name} was detected.\n"
                f"- The observed difference could be explained by random fluctuations.\n"
                f"- There is insufficient evidence to conclude that the {intervention_name} had "
                f"any causal effect on {target_name}.\n\n"
                f"Recommendation: Further analysis or a longer observation period may be needed "
                f"to detect potential effects of the {intervention_name}.\n"
            )
        
        return summary
    
    @staticmethod
    def generate_html_report(
        model_results: Dict,
        metrics: Dict,
        intervention_name: str = "intervention",
        target_name: str = "target variable",
    ) -> str:
        """
        Generate an HTML report of the causal impact analysis.
        
        Args:
            model_results: Dictionary of model results
            metrics: Dictionary of calculated metrics
            intervention_name: Name of the intervention
            target_name: Name of the target variable
            
        Returns:
            HTML report as string
        """
        # Extract key metrics
        rel_effect = model_results.get("relative_effect", 0) * 100
        abs_effect = model_results.get("average_effect", 0)
        cum_effect = model_results.get("cumulative_effect", 0)
        p_value = model_results.get("p_value", 1)
        
        # Determine the direction and significance
        if rel_effect > 0:
            direction = "positive"
            verb = "increased"
            color = "green"
        else:
            direction = "negative"
            verb = "decreased"
            color = "red"
            
        significant = p_value < 0.05
        
        # Generate HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Causal Impact Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                .summary {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
                .significant {{
                    font-weight: bold;
                }}
                .not-significant {{
                    color: #777;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <h1>Causal Impact Analysis Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>
                    The {intervention_name} had a <span class="{direction} {'significant' if significant else 'not-significant'}">{direction}</span> 
                    effect on {target_name}, which {verb} by <span class="{direction}">{abs(rel_effect):.1f}%</span>.
                    This effect is statistically {'significant' if significant else 'not significant'} (p = {p_value:.3f}).
                </p>
            </div>
            
            <h2>Effect Details</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Relative Effect</td>
                    <td class="{color}">{rel_effect:.2f}%</td>
                </tr>
                <tr>
                    <td>Absolute Effect</td>
                    <td>{abs_effect:.4f}</td>
                </tr>
                <tr>
                    <td>Cumulative Effect</td>
                    <td>{cum_effect:.4f}</td>
                </tr>
                <tr>
                    <td>95% CI (Relative)</td>
                    <td>[{model_results.get('relative_effect_lower', 0) * 100:.2f}%, {model_results.get('relative_effect_upper', 0) * 100:.2f}%]</td>
                </tr>
                <tr>
                    <td>p-value</td>
                    <td class="{'significant' if significant else 'not-significant'}">{p_value:.4f}</td>
                </tr>
                <tr>
                    <td>Posterior Probability</td>
                    <td>{model_results.get('posterior_probability', 0) * 100:.1f}%</td>
                </tr>
            </table>
            
            <h2>Model Fit Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>RMSE (Pre-Period)</td>
                    <td>{metrics.get('pre_rmse', 0):.4f}</td>
                </tr>
                <tr>
                    <td>MAE (Pre-Period)</td>
                    <td>{metrics.get('pre_mae', 0):.4f}</td>
                </tr>
                <tr>
                    <td>R-squared (Pre-Period)</td>
                    <td>{metrics.get('pre_r_squared', 0):.4f}</td>
                </tr>
        """
        
        # Add MAPE if available
        if "pre_mape" in metrics:
            html += f"""
                <tr>
                    <td>MAPE (Pre-Period)</td>
                    <td>{metrics.get('pre_mape', 0):.4f}%</td>
                </tr>
            """
            
        html += f"""
            </table>
            
            <h2>Statistical Significance</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>t-statistic</td>
                    <td>{metrics.get('t_statistic', 0):.4f}</td>
                </tr>
                <tr>
                    <td>z-score</td>
                    <td>{metrics.get('z_score', 0):.4f}</td>
                </tr>
                <tr>
                    <td>Cohen's d</td>
                    <td>{metrics.get('cohens_d', 0):.4f}</td>
                </tr>
            </table>
            
            <h2>Interpretation</h2>
            <p>
        """
        
        if significant:
            if rel_effect > 0:
                html += f"""
                The analysis provides strong evidence that the {intervention_name} caused a positive effect 
                on {target_name}. The {target_name} increased by approximately {abs(rel_effect):.1f}% 
                compared to what would have happened without the {intervention_name}.
                """
            else:
                html += f"""
                The analysis provides strong evidence that the {intervention_name} caused a negative effect 
                on {target_name}. The {target_name} decreased by approximately {abs(rel_effect):.1f}% 
                compared to what would have happened without the {intervention_name}.
                """
        else:
            html += f"""
            The analysis does not provide sufficient evidence to conclude that the {intervention_name} 
            had a causal effect on {target_name}. The observed difference of {abs(rel_effect):.1f}% 
            could be explained by random fluctuations.
            """
            
        html += """
            </p>
        </body>
        </html>
        """
        
        return html 