"""
Base model for causal inference models.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class BaseCausalModel(ABC):
    """Abstract base class for all causal inference models."""

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
        Initialize the base causal model.
        
        Args:
            data: DataFrame containing time series data
            pre_period: List of two dates defining pre-intervention period [start, end]
            post_period: List of two dates defining post-intervention period [start, end]
            target_col: Column name of the target variable
            date_col: Column name of the date variable
            covariates: List of column names to use as covariates/controls
            model_args: Additional model-specific arguments
        """
        self.data = data.copy()
        self.pre_period = pre_period
        self.post_period = post_period
        self.target_col = target_col
        self.date_col = date_col
        self.covariates = covariates if covariates else []
        self.model_args = model_args if model_args else {}
        
        # Results storage
        self.results = None
        self.model = None
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self) -> None:
        """Validate the input data and parameters."""
        # Check if required columns exist
        if self.target_col not in self.data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")
            
        if self.date_col not in self.data.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in data")
            
        # Check if covariates exist
        for cov in self.covariates:
            if cov not in self.data.columns:
                raise ValueError(f"Covariate '{cov}' not found in data")
                
        # Ensure date column is datetime
        try:
            self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])
        except Exception as e:
            raise ValueError(f"Could not convert {self.date_col} to datetime: {e}")
            
        # Check pre and post periods
        try:
            pre_start = pd.to_datetime(self.pre_period[0])
            pre_end = pd.to_datetime(self.pre_period[1])
            post_start = pd.to_datetime(self.post_period[0])
            post_end = pd.to_datetime(self.post_period[1])
            
            if pre_start > pre_end:
                raise ValueError("Pre-period start date must be before end date")
                
            if post_start > post_end:
                raise ValueError("Post-period start date must be before end date")
                
            if pre_end >= post_start:
                raise ValueError("Pre-period must end before post-period starts")
                
        except IndexError:
            raise ValueError("Pre and post periods must be lists of two dates [start, end]")
            
        # Check if data covers the entire period
        min_date = self.data[self.date_col].min()
        max_date = self.data[self.date_col].max()
        
        if min_date > pd.to_datetime(self.pre_period[0]) or max_date < pd.to_datetime(self.post_period[1]):
            raise ValueError("Data does not cover the entire pre and post periods")
    
    @abstractmethod
    def fit(self) -> None:
        """Fit the causal impact model."""
        pass
    
    @abstractmethod
    def predict(self) -> pd.DataFrame:
        """Generate counterfactual predictions."""
        pass
    
    @abstractmethod
    def run_inference(self) -> "BaseCausalModel":
        """Run the full inference pipeline and return self for chaining."""
        pass
    
    @abstractmethod
    def get_summary(self) -> Dict:
        """Return a summary of the causal impact results."""
        pass
    
    @abstractmethod
    def plot_results(self) -> None:
        """Plot the results of the causal impact analysis."""
        pass
    
    def get_cumulative_effect(self) -> float:
        """
        Calculate the cumulative effect of the intervention.
        
        Returns:
            The cumulative effect (absolute difference between actual and counterfactual)
        """
        if self.results is None:
            raise ValueError("Model has not been run yet. Call run_inference() first.")
            
        return self.results["cumulative_effect"]
    
    def get_relative_effect(self) -> float:
        """
        Calculate the relative effect of the intervention as a percentage.
        
        Returns:
            The relative effect as a percentage
        """
        if self.results is None:
            raise ValueError("Model has not been run yet. Call run_inference() first.")
            
        return self.results["relative_effect"]
    
    def get_posterior_probability(self) -> float:
        """
        Get the posterior probability that the effect is causal.
        
        Returns:
            Posterior probability (0 to 1)
        """
        if self.results is None:
            raise ValueError("Model has not been run yet. Call run_inference() first.")
            
        return self.results["posterior_probability"]
    
    def get_significance(self, alpha: float = 0.05) -> bool:
        """
        Determine if the causal effect is statistically significant.
        
        Args:
            alpha: Significance level (default: 0.05)
            
        Returns:
            True if effect is significant, False otherwise
        """
        if self.results is None:
            raise ValueError("Model has not been run yet. Call run_inference() first.")
            
        return self.results["p_value"] < alpha 