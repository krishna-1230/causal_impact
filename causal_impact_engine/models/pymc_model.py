"""
PyMC-based implementation of Bayesian structural time series for causal inference.
"""
from typing import Dict, List, Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from scipy import stats

from causal_impact_engine.models.base_model import BaseCausalModel


class PyMCCausalModel(BaseCausalModel):
    """
    Causal inference model using PyMC for Bayesian structural time series.
    
    This model implements a custom Bayesian approach to causal inference
    using PyMC's probabilistic programming capabilities. It provides more
    flexibility than the CausalImpact package but requires more careful
    model specification.
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
        Initialize the PyMC causal model.
        
        Args:
            data: DataFrame containing time series data
            pre_period: List of two dates defining pre-intervention period [start, end]
            post_period: List of two dates defining post-intervention period [start, end]
            target_col: Column name of the target variable
            date_col: Column name of the date variable
            covariates: List of column names to use as covariates/controls
            model_args: Additional model-specific arguments
                - num_samples: Number of MCMC samples (default: 2000)
                - chains: Number of MCMC chains (default: 4)
                - tune: Number of tuning steps (default: 1000)
                - seasonal_period: Period for seasonality component (default: None)
                - include_trend: Whether to include trend component (default: True)
                - standardize: Whether to standardize variables (default: True)
        """
        super().__init__(data, pre_period, post_period, target_col, date_col, covariates, model_args)
        
        # Set default model arguments if not provided
        self._set_default_args()
        
        # Prepare data for PyMC
        self._prepare_data()
        
    def _set_default_args(self) -> None:
        """Set default model arguments if not provided."""
        defaults = {
            "num_samples": 2000,
            "chains": 4,
            "tune": 1000,
            "seasonal_period": None,
            "include_trend": True,
            "standardize": True
        }
        
        for key, value in defaults.items():
            if key not in self.model_args:
                self.model_args[key] = value
    
    def _prepare_data(self) -> None:
        """Prepare the data for PyMC modeling."""
        # Set date as index
        self.formatted_data = self.data.copy()
        self.formatted_data = self.formatted_data.set_index(self.date_col)
        
        # Convert pre_period and post_period to datetime
        self.pre_period_dt = [pd.to_datetime(date) for date in self.pre_period]
        self.post_period_dt = [pd.to_datetime(date) for date in self.post_period]
        
        # Create intervention indicator
        self.formatted_data["intervention"] = 0
        intervention_start = self.post_period_dt[0]
        self.formatted_data.loc[self.formatted_data.index >= intervention_start, "intervention"] = 1
        
        # Create time index
        self.formatted_data["time_idx"] = np.arange(len(self.formatted_data))
        
        # Split data into pre and post periods
        self.pre_data = self.formatted_data[
            (self.formatted_data.index >= self.pre_period_dt[0]) & 
            (self.formatted_data.index <= self.pre_period_dt[1])
        ]
        self.post_data = self.formatted_data[
            (self.formatted_data.index >= self.post_period_dt[0]) & 
            (self.formatted_data.index <= self.post_period_dt[1])
        ]
        
        # Standardize variables if requested
        if self.model_args["standardize"]:
            # Calculate mean and std from pre-intervention period
            self.target_mean = self.pre_data[self.target_col].mean()
            self.target_std = self.pre_data[self.target_col].std()
            
            # Standardize target
            self.formatted_data[f"{self.target_col}_scaled"] = (
                (self.formatted_data[self.target_col] - self.target_mean) / self.target_std
            )
            
            # Standardize covariates
            self.covariate_means = {}
            self.covariate_stds = {}
            for cov in self.covariates:
                self.covariate_means[cov] = self.pre_data[cov].mean()
                self.covariate_stds[cov] = self.pre_data[cov].std()
                
                self.formatted_data[f"{cov}_scaled"] = (
                    (self.formatted_data[cov] - self.covariate_means[cov]) / self.covariate_stds[cov]
                )
            
            # Update column names for scaled data
            self.target_col_model = f"{self.target_col}_scaled"
            self.covariates_model = [f"{cov}_scaled" for cov in self.covariates]
        else:
            self.target_col_model = self.target_col
            self.covariates_model = self.covariates
    
    def fit(self) -> None:
        """Fit the PyMC causal impact model."""
        # Create PyMC model
        with pm.Model() as self.model:
            # Time indices
            time = self.formatted_data["time_idx"].values
            intervention = self.formatted_data["intervention"].values
            
            # Data arrays
            y_observed = self.formatted_data[self.target_col_model].values
            
            # Trend component
            if self.model_args["include_trend"]:
                trend_sigma = pm.HalfCauchy("trend_sigma", beta=0.1)
                trend = pm.GaussianRandomWalk(
                    "trend", 
                    sigma=trend_sigma, 
                    shape=len(time)
                )
            else:
                trend = 0
            
            # Seasonal component (if requested)
            if self.model_args["seasonal_period"] is not None:
                period = self.model_args["seasonal_period"]
                seasonal_sigma = pm.HalfCauchy("seasonal_sigma", beta=0.1)
                seasonal = pm.MvNormal(
                    "seasonal",
                    mu=0,
                    cov=seasonal_sigma * np.eye(period),
                    shape=(len(time) // period + 1, period)
                )
                seasonal_component = seasonal[time // period, time % period]
            else:
                seasonal_component = 0
            
            # Regression component for covariates
            if self.covariates:
                X = np.column_stack([self.formatted_data[cov].values for cov in self.covariates_model])
                beta = pm.Normal("beta", mu=0, sigma=1, shape=len(self.covariates))
                regression = pm.math.dot(X, beta)
            else:
                regression = 0
            
            # Intervention effect
            impact = pm.Normal("impact", mu=0, sigma=1)
            intervention_effect = impact * intervention
            
            # Observation noise
            sigma = pm.HalfCauchy("sigma", beta=1)
            
            # Expected value
            mu = trend + seasonal_component + regression + intervention_effect
            
            # Likelihood
            y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_observed)
            
            # Sample from posterior
            self.trace = pm.sample(
                draws=self.model_args["num_samples"],
                chains=self.model_args["chains"],
                tune=self.model_args["tune"],
                return_inferencedata=True
            )
    
    def predict(self) -> pd.DataFrame:
        """
        Generate counterfactual predictions.
        
        Returns:
            DataFrame with actual values, counterfactual predictions, and credible intervals
        """
        if self.trace is None:
            raise ValueError("Model has not been fit yet. Call run_inference() first.")
        
        # Extract posterior samples
        impact_samples = self.trace.posterior["impact"].values.flatten()
        
        # Create counterfactual predictions (without intervention)
        counterfactual = self.formatted_data.copy()
        
        # Calculate counterfactual values
        counterfactual["prediction"] = counterfactual[self.target_col]
        counterfactual.loc[counterfactual["intervention"] == 1, "prediction"] -= np.mean(impact_samples)
        
        # Calculate point-wise effect
        counterfactual["effect"] = 0
        counterfactual.loc[counterfactual["intervention"] == 1, "effect"] = (
            counterfactual.loc[counterfactual["intervention"] == 1, self.target_col] - 
            counterfactual.loc[counterfactual["intervention"] == 1, "prediction"]
        )
        
        # Calculate credible intervals for the counterfactual
        impact_lower = np.percentile(impact_samples, 2.5)
        impact_upper = np.percentile(impact_samples, 97.5)
        
        counterfactual["prediction_lower"] = counterfactual["prediction"]
        counterfactual["prediction_upper"] = counterfactual["prediction"]
        
        # Add credible intervals for the post-intervention period
        counterfactual.loc[counterfactual["intervention"] == 1, "prediction_lower"] = (
            counterfactual.loc[counterfactual["intervention"] == 1, self.target_col] - impact_upper
        )
        counterfactual.loc[counterfactual["intervention"] == 1, "prediction_upper"] = (
            counterfactual.loc[counterfactual["intervention"] == 1, self.target_col] - impact_lower
        )
        
        # Store predictions
        self.predictions = counterfactual
        
        return counterfactual
    
    def run_inference(self) -> "PyMCCausalModel":
        """
        Run the full causal inference pipeline.
        
        Returns:
            Self for method chaining
        """
        # Fit the model
        self.fit()
        
        # Generate predictions
        self.predict()
        
        # Extract results
        self._extract_results()
        
        return self
    
    def _extract_results(self) -> None:
        """Extract and format results from the PyMC model."""
        if self.trace is None:
            raise ValueError("Model has not been fit yet. Call run_inference() first.")
        
        # Extract impact samples
        impact_samples = self.trace.posterior["impact"].values.flatten()
        
        # Calculate summary statistics
        mean_effect = np.mean(impact_samples)
        if self.model_args["standardize"]:
            # Convert back to original scale
            mean_effect = mean_effect * self.target_std
            
        # Calculate credible intervals
        ci_lower = np.percentile(impact_samples, 2.5)
        ci_upper = np.percentile(impact_samples, 97.5)
        
        if self.model_args["standardize"]:
            ci_lower = ci_lower * self.target_std
            ci_upper = ci_upper * self.target_std
            
        # Calculate cumulative effect
        post_intervention = self.formatted_data[self.formatted_data["intervention"] == 1]
        n_post = len(post_intervention)
        cumulative_effect = mean_effect * n_post
        
        # Calculate relative effect
        pre_intervention = self.formatted_data[self.formatted_data["intervention"] == 0]
        pre_mean = pre_intervention[self.target_col].mean()
        relative_effect = mean_effect / pre_mean
        
        # Calculate posterior probability
        posterior_probability = np.mean(impact_samples > 0)
        p_value = min(posterior_probability, 1 - posterior_probability) * 2
        
        # Store results
        self.results = {
            "mean_effect": mean_effect,
            "effect_lower": ci_lower,
            "effect_upper": ci_upper,
            "cumulative_effect": cumulative_effect,
            "cumulative_effect_lower": ci_lower * n_post,
            "cumulative_effect_upper": ci_upper * n_post,
            "relative_effect": relative_effect,
            "relative_effect_lower": ci_lower / pre_mean,
            "relative_effect_upper": ci_upper / pre_mean,
            "posterior_probability": posterior_probability,
            "p_value": p_value,
            "impact_samples": impact_samples
        }
    
    def get_summary(self) -> Dict:
        """
        Return a summary of the causal impact results.
        
        Returns:
            Dictionary containing summary statistics
        """
        if self.results is None:
            raise ValueError("Model has not been run yet. Call run_inference() first.")
            
        # Generate textual report
        report = self._generate_report()
            
        return {
            "average_effect": self.results["mean_effect"],
            "average_effect_lower": self.results["effect_lower"],
            "average_effect_upper": self.results["effect_upper"],
            "cumulative_effect": self.results["cumulative_effect"],
            "cumulative_effect_lower": self.results["cumulative_effect_lower"],
            "cumulative_effect_upper": self.results["cumulative_effect_upper"],
            "relative_effect": self.results["relative_effect"],
            "relative_effect_lower": self.results["relative_effect_lower"],
            "relative_effect_upper": self.results["relative_effect_upper"],
            "posterior_probability": self.results["posterior_probability"],
            "p_value": self.results["p_value"],
            "is_significant": self.results["p_value"] < 0.05,
            "report": report
        }
    
    def _generate_report(self) -> str:
        """Generate a textual report of the causal impact analysis."""
        if self.results is None:
            raise ValueError("Model has not been run yet. Call run_inference() first.")
            
        # Format numbers for report
        rel_effect = self.results["relative_effect"] * 100
        prob = self.results["posterior_probability"] * 100
        
        # Determine the direction of the effect
        if rel_effect > 0:
            direction = "positive"
            verb = "increased"
        else:
            direction = "negative"
            verb = "decreased"
            
        # Format confidence interval
        ci_lower = self.results["relative_effect_lower"] * 100
        ci_upper = self.results["relative_effect_upper"] * 100
        
        # Generate report
        report = (
            f"Causal Impact Analysis Report\n"
            f"==============================\n\n"
            f"During the post-intervention period, the response variable {verb} by "
            f"{abs(rel_effect):.1f}% ({ci_lower:.1f}% to {ci_upper:.1f}%).\n\n"
            f"The probability of observing this effect by chance is p = {self.results['p_value']:.3f}. "
            f"This means that the causal effect can be considered statistically "
        )
        
        if self.results["p_value"] < 0.05:
            report += "significant.\n\n"
        else:
            report += "insignificant.\n\n"
            
        report += (
            f"The probability that the effect is {direction} is {prob:.1f}%.\n\n"
            f"Posterior inference was done using PyMC with {self.model_args['num_samples']} MCMC samples "
            f"and {self.model_args['chains']} chains.\n"
        )
        
        return report
    
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
            
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot original and counterfactual
        ax1 = axes[0]
        self.predictions.plot(y=self.target_col, ax=ax1, label="Observed")
        self.predictions.plot(y="prediction", ax=ax1, label="Counterfactual", linestyle="--")
        ax1.fill_between(
            self.predictions.index,
            self.predictions["prediction_lower"],
            self.predictions["prediction_upper"],
            alpha=0.2,
            color="g"
        )
        
        # Add intervention line
        intervention_start = self.post_period_dt[0]
        ax1.axvline(intervention_start, color="r", linestyle="--", label="Intervention")
        ax1.set_title("Actual vs Counterfactual")
        ax1.set_ylabel(self.target_col)
        ax1.legend()
        
        # Plot pointwise effects
        ax2 = axes[1]
        self.predictions.plot(y="effect", ax=ax2)
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.2)
        ax2.axvline(intervention_start, color="r", linestyle="--")
        ax2.set_title("Pointwise Effects")
        ax2.set_ylabel("Effect")
        
        # Plot cumulative effects
        ax3 = axes[2]
        cumulative = self.predictions["effect"].cumsum()
        cumulative.plot(ax=ax3)
        ax3.axhline(y=0, color="k", linestyle="-", alpha=0.2)
        ax3.axvline(intervention_start, color="r", linestyle="--")
        ax3.set_title("Cumulative Effect")
        ax3.set_ylabel("Cumulative Effect")
        ax3.set_xlabel("")
        
        # Add a title with summary information
        effect = self.results["relative_effect"] * 100
        prob = self.results["posterior_probability"] * 100
        plt.suptitle(
            f"Causal Impact Analysis (PyMC)\n"
            f"Relative Effect: {effect:.1f}% ({self.results['p_value']:.3f})\n"
            f"Posterior Probability: {prob:.1f}%",
            fontsize=16
        )
        
        plt.tight_layout()
        return fig
    
    def plot_posterior(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the posterior distribution of the intervention effect.
        
        Args:
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure object
        """
        if self.results is None:
            raise ValueError("Model has not been run yet. Call run_inference() first.")
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot posterior distribution
        impact_samples = self.results["impact_samples"]
        if self.model_args["standardize"]:
            impact_samples = impact_samples * self.target_std
            
        sns.histplot(impact_samples, kde=True, ax=ax)
        
        # Add vertical lines for mean and CI
        ax.axvline(self.results["mean_effect"], color="r", linestyle="-", label="Mean")
        ax.axvline(self.results["effect_lower"], color="r", linestyle="--", label="95% CI")
        ax.axvline(self.results["effect_upper"], color="r", linestyle="--")
        
        # Add zero line
        ax.axvline(0, color="k", linestyle="-", alpha=0.2, label="No Effect")
        
        # Set labels and title
        ax.set_xlabel("Effect Size")
        ax.set_ylabel("Density")
        ax.set_title(f"Posterior Distribution of Causal Effect\nP(effect > 0) = {self.results['posterior_probability']:.3f}")
        ax.legend()
        
        return fig
    
    def get_formatted_report(self) -> str:
        """
        Get a formatted text report of the causal impact analysis.
        
        Returns:
            Formatted report as string
        """
        if self.results is None:
            raise ValueError("Model has not been run yet. Call run_inference() first.")
            
        return self._generate_report() 