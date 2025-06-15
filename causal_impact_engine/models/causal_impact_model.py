"""
Implementation of Google's CausalImpact package for causal inference.
"""
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tools import diff

from causal_impact_engine.models.base_model import BaseCausalModel


class CausalImpactModel(BaseCausalModel):
    """
    Causal inference model using Bayesian structural time series approach.
    
    This model implements a simplified version of the Bayesian structural time series approach
    to causal inference as described in:
    
    Brodersen et al., Annals of Applied Statistics (2015)
    http://research.google.com/pubs/pub41854.html
    """

    def __init__(
        self,
        data: pd.DataFrame,
        pre_period: List[str],
        post_period: List[str],
        target_col: str = "y",
        date_col: str = "date",
        covariates: Optional[List[str]] = None,
        model_args: Optional[Dict] = None,
    ):
        """
        Initialize the CausalImpact model.
        
        Args:
            data: DataFrame containing time series data
            pre_period: List of two dates defining pre-intervention period [start, end]
            post_period: List of two dates defining post-intervention period [start, end]
            target_col: Column name of the target variable
            date_col: Column name of the date variable
            covariates: List of column names to use as covariates/controls
            model_args: Additional model-specific arguments for CausalImpact
        """
        super().__init__(data, pre_period, post_period, target_col, date_col, covariates, model_args)
        
        # Format data for analysis
        self._prepare_data()
        
    def _prepare_data(self) -> None:
        """Prepare the data for analysis."""
        # Set date as index
        self.formatted_data = self.data.copy()
        if self.date_col in self.formatted_data.columns:
            self.formatted_data = self.formatted_data.set_index(self.date_col)
        
        # Select only target and covariates
        cols_to_use = [self.target_col] + self.covariates
        self.formatted_data = self.formatted_data[cols_to_use]
        
        # Convert pre_period and post_period to datetime
        self.pre_period_dt = [pd.to_datetime(date) for date in self.pre_period]
        self.post_period_dt = [pd.to_datetime(date) for date in self.post_period]
        
        # Create pre and post masks
        self.pre_mask = (self.formatted_data.index >= self.pre_period_dt[0]) & (self.formatted_data.index <= self.pre_period_dt[1])
        self.post_mask = (self.formatted_data.index >= self.post_period_dt[0]) & (self.formatted_data.index <= self.post_period_dt[1])
    
    def fit(self) -> None:
        """Fit the model to the pre-intervention data."""
        # Get pre-intervention data
        pre_data = self.formatted_data[self.pre_mask].copy()
        
        # Fit a structural time series model to the pre-intervention data
        y_pre = pre_data[self.target_col]
        
        # Create design matrix with covariates
        X = None
        if self.covariates:
            X = pre_data[self.covariates]
            # Add constant
            X = pd.DataFrame({'const': 1, **X})
        
        # Fit model
        if X is not None:
            # Use exogenous variables if available
            self.model = UnobservedComponents(
                y_pre, 
                level='local linear trend', 
                seasonal=7,  # Weekly seasonality
                exog=X
            )
        else:
            # Simple model without exogenous variables
            self.model = UnobservedComponents(
                y_pre, 
                level='local linear trend',
                seasonal=7  # Weekly seasonality
            )
            
        self.fit_results = self.model.fit()
    
    def predict(self) -> pd.DataFrame:
        """Generate counterfactual predictions."""
        if self.model is None or self.fit_results is None:
            raise ValueError("Model has not been fit yet. Call run_inference() first.")
            
        # Create full dataset for prediction
        full_data = self.formatted_data.copy()
        
        # Get pre-intervention data for reference
        pre_data = full_data[self.pre_mask].copy()
        
        # Create predictions for pre-intervention period first
        if self.covariates:
            # Get pre-intervention exogenous variables
            X_pre = pre_data[self.covariates]
            X_pre = pd.DataFrame({'const': 1, **X_pre})
            
            # In-sample predictions
            pred_pre = self.fit_results.predict()
        else:
            # Simple model without exogenous variables
            pred_pre = self.fit_results.predict()
        
        # Create post-intervention predictions
        post_data = full_data[self.post_mask].copy()
        
        if self.covariates:
            # Get post-intervention exogenous variables
            X_post = post_data[self.covariates]
            X_post = pd.DataFrame({'const': 1, **X_post})
            
            # Out-of-sample forecast
            steps = len(post_data)
            pred_post = self.fit_results.forecast(steps=steps, exog=X_post)
        else:
            # Simple model without exogenous variables
            steps = len(post_data)
            pred_post = self.fit_results.forecast(steps=steps)
        
        # Combine predictions
        full_data.loc[self.pre_mask, 'prediction'] = pred_pre
        full_data.loc[self.post_mask, 'prediction'] = pred_post
        
        # Calculate confidence intervals (simple approximation)
        # Get standard deviation of residuals for confidence intervals
        residuals = pre_data[self.target_col] - pred_pre
        sigma = np.std(residuals)
        
        full_data['prediction_lower'] = full_data['prediction'] - 1.96 * sigma
        full_data['prediction_upper'] = full_data['prediction'] + 1.96 * sigma
        
        # Calculate point-wise effect (actual - predicted)
        full_data['effect'] = full_data[self.target_col] - full_data['prediction']
        
        # Add intervention indicator
        full_data['intervention'] = 0
        full_data.loc[self.post_mask, 'intervention'] = 1
        
        # Reset index to get date as column
        if self.date_col not in full_data.columns:
            full_data = full_data.reset_index()
            
        # Store predictions
        self.predictions = full_data
        
        return full_data
    
    def run_inference(self) -> "CausalImpactModel":
        """
        Run the full causal inference pipeline.
        
        Returns:
            Self for method chaining
        """
        try:
            # Fit model on pre-intervention data
            self.fit()
            
            # Generate predictions for full period
            self.predict()
            
            # Extract results
            self._extract_results()
        except Exception as e:
            print(f"Error in CausalImpact: {e}")
            # Create a minimal model with default values
            self.results = {
                "cumulative_effect": 0.0,
                "cumulative_effect_lower": 0.0,
                "cumulative_effect_upper": 0.0,
                "relative_effect": 0.0,
                "relative_effect_lower": 0.0,
                "relative_effect_upper": 0.0,
                "posterior_probability": 0.5,
                "p_value": 0.5,
                "model_summary": f"CausalImpact failed: {str(e)}",
                "report": f"CausalImpact failed: {str(e)}",
                "inferences": pd.DataFrame({
                    'point_effect': [0.0],
                    'point_effect_lower': [0.0],
                    'point_effect_upper': [0.0],
                    'cum_effect': [0.0],
                    'cum_effect_lower': [0.0],
                    'cum_effect_upper': [0.0],
                })
            }
            
            # Create empty predictions
            self.predictions = pd.DataFrame()
        
        return self
    
    def _extract_results(self) -> None:
        """Extract and format results from the model."""
        if self.predictions is None or self.predictions.empty:
            raise ValueError("Model has not been run yet. Call run_inference() first.")
        
        # Extract post-intervention data
        post_data = self.predictions[self.post_mask].copy()
        
        # Calculate pre-intervention mean
        pre_data = self.predictions[self.pre_mask]
        pre_mean = pre_data[self.target_col].mean()
        
        # Calculate effects
        point_effects = post_data['effect']
        cum_effect = point_effects.sum()
        rel_effect = cum_effect / (pre_mean * len(point_effects))
        
        # Calculate standard errors and confidence intervals
        effect_std = point_effects.std()
        effect_lower = cum_effect - 1.96 * effect_std * np.sqrt(len(point_effects))
        effect_upper = cum_effect + 1.96 * effect_std * np.sqrt(len(point_effects))
        
        rel_effect_lower = effect_lower / (pre_mean * len(point_effects))
        rel_effect_upper = effect_upper / (pre_mean * len(point_effects))
        
        # Calculate p-value (simple approximation)
        z_score = cum_effect / (effect_std * np.sqrt(len(point_effects)))
        p_value = 2 * (1 - abs(np.clip(z_score / np.sqrt(2), -0.99, 0.99)))
        
        # Create inferences DataFrame
        inferences = self.predictions.copy()
        inferences['point_effect'] = 0
        inferences['point_effect_lower'] = 0
        inferences['point_effect_upper'] = 0
        inferences['cum_effect'] = 0
        inferences['cum_effect_lower'] = 0
        inferences['cum_effect_upper'] = 0
        
        # Fill post-intervention values
        inferences.loc[self.post_mask, 'point_effect'] = point_effects
        inferences.loc[self.post_mask, 'point_effect_lower'] = point_effects - 1.96 * effect_std
        inferences.loc[self.post_mask, 'point_effect_upper'] = point_effects + 1.96 * effect_std
        
        # Calculate cumulative effects
        cum_effects = point_effects.cumsum()
        inferences.loc[self.post_mask, 'cum_effect'] = cum_effects
        inferences.loc[self.post_mask, 'cum_effect_lower'] = cum_effects - 1.96 * effect_std * np.sqrt(np.arange(1, len(cum_effects) + 1))
        inferences.loc[self.post_mask, 'cum_effect_upper'] = cum_effects + 1.96 * effect_std * np.sqrt(np.arange(1, len(cum_effects) + 1))
        
        # Generate report
        report = self._generate_report(cum_effect, rel_effect, p_value)
        
        # Store results
        self.results = {
            "cumulative_effect": float(cum_effect),
            "cumulative_effect_lower": float(effect_lower),
            "cumulative_effect_upper": float(effect_upper),
            "relative_effect": float(rel_effect),
            "relative_effect_lower": float(rel_effect_lower),
            "relative_effect_upper": float(rel_effect_upper),
            "posterior_probability": 1 - float(p_value),
            "p_value": float(p_value),
            "model_summary": report,
            "report": report,
            "inferences": inferences
        }
    
    def _generate_report(self, cum_effect, rel_effect, p_value):
        """Generate a report similar to CausalImpact's summary."""
        rel_effect_pct = rel_effect * 100
        
        if rel_effect > 0:
            direction = "positive"
            verb = "increased"
        else:
            direction = "negative"
            verb = "decreased"
            rel_effect_pct = abs(rel_effect_pct)
        
        report = (
            f"Causal Impact Analysis\n"
            f"======================\n\n"
            f"During the post-intervention period, the response variable {verb}\n"
            f"by {rel_effect_pct:.2f}%.\n\n"
            f"Cumulative effect: {cum_effect:.2f}\n"
            f"Relative effect: {rel_effect_pct:.2f}%\n\n"
            f"The probability of obtaining this effect by chance is p = {p_value:.4f}.\n"
            f"The causal effect can be considered statistically {'significant' if p_value < 0.05 else 'not significant'}."
        )
        
        return report
    
    def get_summary(self) -> Dict:
        """
        Return a summary of the causal impact results.
        
        Returns:
            Dictionary containing summary statistics
        """
        if self.results is None:
            raise ValueError("Model has not been run yet. Call run_inference() first.")
        
        # Calculate average effect safely
        avg_effect = 0.0
        avg_effect_lower = 0.0
        avg_effect_upper = 0.0
        
        try:
            # Check if we have valid predictions and post_mask
            if hasattr(self, 'predictions') and self.predictions is not None and not self.predictions.empty:
                # Count post-intervention periods
                if hasattr(self, 'post_mask') and self.post_mask is not None:
                    post_periods = self.predictions[self.post_mask]
                    if not post_periods.empty:
                        n_periods = len(post_periods)
                        if n_periods > 0:
                            avg_effect = self.results["cumulative_effect"] / n_periods
                            avg_effect_lower = self.results["cumulative_effect_lower"] / n_periods
                            avg_effect_upper = self.results["cumulative_effect_upper"] / n_periods
        except Exception as e:
            print(f"Warning: Error calculating average effects: {e}")
            # Use cumulative effect as fallback
            avg_effect = self.results["cumulative_effect"]
            avg_effect_lower = self.results["cumulative_effect_lower"]
            avg_effect_upper = self.results["cumulative_effect_upper"]
            
        return {
            "average_effect": avg_effect,
            "average_effect_lower": avg_effect_lower,
            "average_effect_upper": avg_effect_upper,
            "cumulative_effect": self.results["cumulative_effect"],
            "cumulative_effect_lower": self.results["cumulative_effect_lower"],
            "cumulative_effect_upper": self.results["cumulative_effect_upper"],
            "relative_effect": self.results["relative_effect"],
            "relative_effect_lower": self.results["relative_effect_lower"],
            "relative_effect_upper": self.results["relative_effect_upper"],
            "posterior_probability": self.results["posterior_probability"],
            "p_value": self.results["p_value"],
            "is_significant": self.results["p_value"] < 0.05,
            "report": self.results["report"]
        }
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Plot the results of the causal impact analysis.
        
        Args:
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure object
        """
        if self.results is None or self.predictions is None:
            raise ValueError("Model has not been run yet. Call run_inference() first.")
            
        # Create a new figure with specified size
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot original vs counterfactual
        ax1 = axes[0]
        self.predictions.plot(y=self.target_col, ax=ax1, label="Actual")
        self.predictions.plot(y="prediction", ax=ax1, label="Counterfactual", style="--")
        ax1.fill_between(
            self.predictions.index,
            self.predictions["prediction_lower"],
            self.predictions["prediction_upper"],
            alpha=0.2,
            label="95% CI"
        )
        ax1.axvline(self.post_period_dt[0], color="red", linestyle="--", label="Intervention")
        ax1.set_title("Original vs Counterfactual")
        ax1.set_ylabel("Value")
        ax1.legend()
        
        # Plot pointwise effect
        ax2 = axes[1]
        self.predictions.plot(y="effect", ax=ax2)
        ax2.axhline(0, color="black", linestyle="--")
        ax2.axvline(self.post_period_dt[0], color="red", linestyle="--")
        ax2.set_title("Pointwise Effect")
        ax2.set_ylabel("Effect")
        
        # Plot cumulative effect
        ax3 = axes[2]
        cum_effect = self.predictions["effect"].cumsum()
        cum_effect.loc[self.predictions.index < self.post_period_dt[0]] = 0
        ax3.plot(self.predictions.index, cum_effect)
        ax3.axhline(0, color="black", linestyle="--")
        ax3.axvline(self.post_period_dt[0], color="red", linestyle="--")
        ax3.set_title("Cumulative Effect")
        ax3.set_ylabel("Cumulative Effect")
        ax3.set_xlabel("Date")
        
        # Add a title with summary information
        effect = self.results["relative_effect"] * 100
        prob = self.results["posterior_probability"] * 100
        plt.suptitle(
            f"Causal Impact Analysis\n"
            f"Relative Effect: {effect:.1f}% (p = {self.results['p_value']:.3f})\n"
            f"Posterior Probability: {prob:.1f}%",
            fontsize=16
        )
        
        plt.tight_layout()
        return fig
    
    def get_formatted_report(self) -> str:
        """
        Get a formatted text report of the causal impact analysis.
        
        Returns:
            Formatted report as string
        """
        if self.results is None:
            raise ValueError("Model has not been run yet. Call run_inference() first.")
            
        return self.results["report"] 