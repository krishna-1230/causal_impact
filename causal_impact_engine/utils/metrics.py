"""
Metrics utilities for evaluating causal impact models.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


class CausalImpactMetrics:
    """Metrics for evaluating causal impact models."""
    
    @staticmethod
    def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            MAPE value
        """
        return np.mean(np.abs((actual - predicted) / actual)) * 100
    
    @staticmethod
    def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            RMSE value
        """
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    @staticmethod
    def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            MAE value
        """
        return np.mean(np.abs(actual - predicted))
    
    @staticmethod
    def calculate_r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate R-squared.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            R-squared value
        """
        ss_total = np.sum((actual - np.mean(actual)) ** 2)
        ss_residual = np.sum((actual - predicted) ** 2)
        return 1 - (ss_residual / ss_total)
    
    @staticmethod
    def calculate_pre_period_fit_metrics(
        data: pd.DataFrame,
        target_col: str = "y",
        prediction_col: str = "prediction",
        intervention_col: str = "intervention",
    ) -> Dict[str, float]:
        """
        Calculate fit metrics for the pre-intervention period.
        
        Args:
            data: DataFrame with actual and predicted values
            target_col: Name of the target column
            prediction_col: Name of the prediction column
            intervention_col: Name of the intervention indicator column
            
        Returns:
            Dictionary of fit metrics
        """
        # Filter pre-intervention period
        pre_data = data[data[intervention_col] == 0]
        
        # Extract actual and predicted values
        actual = pre_data[target_col].values
        predicted = pre_data[prediction_col].values
        
        # Calculate metrics
        metrics = {
            "pre_rmse": CausalImpactMetrics.calculate_rmse(actual, predicted),
            "pre_mae": CausalImpactMetrics.calculate_mae(actual, predicted),
            "pre_r_squared": CausalImpactMetrics.calculate_r_squared(actual, predicted)
        }
        
        # Add MAPE if there are no zeros in actual values
        if not np.any(actual == 0):
            metrics["pre_mape"] = CausalImpactMetrics.calculate_mape(actual, predicted)
            
        return metrics
    
    @staticmethod
    def calculate_effect_size(
        data: pd.DataFrame,
        target_col: str = "y",
        prediction_col: str = "prediction",
        intervention_col: str = "intervention",
    ) -> Dict[str, float]:
        """
        Calculate effect size metrics.
        
        Args:
            data: DataFrame with actual and predicted values
            target_col: Name of the target column
            prediction_col: Name of the prediction column
            intervention_col: Name of the intervention indicator column
            
        Returns:
            Dictionary of effect size metrics
        """
        # Filter pre and post-intervention periods
        pre_data = data[data[intervention_col] == 0]
        post_data = data[data[intervention_col] == 1]
        
        # Calculate mean values
        pre_mean = pre_data[target_col].mean()
        post_mean = post_data[target_col].mean()
        post_counterfactual_mean = post_data[prediction_col].mean()
        
        # Calculate absolute effect
        absolute_effect = post_mean - post_counterfactual_mean
        
        # Calculate relative effect
        relative_effect = absolute_effect / post_counterfactual_mean
        
        # Calculate cumulative effect
        cumulative_effect = absolute_effect * len(post_data)
        
        # Calculate Cohen's d
        # (difference in means divided by pooled standard deviation)
        pooled_std = np.sqrt(
            ((len(post_data) - 1) * post_data[target_col].std() ** 2 + 
             (len(post_data) - 1) * post_data[prediction_col].std() ** 2) / 
            (len(post_data) + len(post_data) - 2)
        )
        cohens_d = absolute_effect / pooled_std if pooled_std != 0 else np.nan
        
        return {
            "absolute_effect": absolute_effect,
            "relative_effect": relative_effect,
            "cumulative_effect": cumulative_effect,
            "cohens_d": cohens_d
        }
    
    @staticmethod
    def calculate_statistical_significance(
        data: pd.DataFrame,
        target_col: str = "y",
        prediction_col: str = "prediction",
        intervention_col: str = "intervention",
    ) -> Dict[str, float]:
        """
        Calculate statistical significance of the effect.
        
        Args:
            data: DataFrame with actual and predicted values
            target_col: Name of the target column
            prediction_col: Name of the prediction column
            intervention_col: Name of the intervention indicator column
            
        Returns:
            Dictionary of statistical significance metrics
        """
        # Filter post-intervention period
        post_data = data[data[intervention_col] == 1]
        
        # Extract actual and predicted values
        actual = post_data[target_col].values
        predicted = post_data[prediction_col].values
        
        # Calculate t-test
        t_stat, p_value = stats.ttest_rel(actual, predicted)
        
        # Calculate z-score
        effect = actual - predicted
        z_score = np.mean(effect) / (np.std(effect) / np.sqrt(len(effect)))
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "z_score": z_score,
            "is_significant": p_value < 0.05
        }
    
    @staticmethod
    def calculate_all_metrics(
        data: pd.DataFrame,
        target_col: str = "y",
        prediction_col: str = "prediction",
        intervention_col: str = "intervention",
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Args:
            data: DataFrame with actual and predicted values
            target_col: Name of the target column
            prediction_col: Name of the prediction column
            intervention_col: Name of the intervention indicator column
            
        Returns:
            Dictionary of all metrics
        """
        # Calculate fit metrics
        fit_metrics = CausalImpactMetrics.calculate_pre_period_fit_metrics(
            data, target_col, prediction_col, intervention_col
        )
        
        # Calculate effect size metrics
        effect_metrics = CausalImpactMetrics.calculate_effect_size(
            data, target_col, prediction_col, intervention_col
        )
        
        # Calculate statistical significance metrics
        significance_metrics = CausalImpactMetrics.calculate_statistical_significance(
            data, target_col, prediction_col, intervention_col
        )
        
        # Combine all metrics
        all_metrics = {}
        all_metrics.update(fit_metrics)
        all_metrics.update(effect_metrics)
        all_metrics.update(significance_metrics)
        
        return all_metrics 