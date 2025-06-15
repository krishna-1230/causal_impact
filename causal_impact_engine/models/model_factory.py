"""
Factory for creating causal inference models.
"""
from typing import Dict, List, Optional, Union

import pandas as pd

from causal_impact_engine.models.base_model import BaseCausalModel
from causal_impact_engine.models.causal_impact_model import CausalImpactModel
from causal_impact_engine.models.pymc_model import PyMCCausalModel


class ModelFactory:
    """Factory class for creating causal inference models."""
    
    @staticmethod
    def create_model(
        model_type: str,
        data: pd.DataFrame,
        pre_period: List[str],
        post_period: List[str],
        target_col: str = "y",
        date_col: str = "date",
        covariates: Optional[List[str]] = None,
        model_args: Optional[Dict] = None,
    ) -> BaseCausalModel:
        """
        Create a causal inference model of the specified type.
        
        Args:
            model_type: Type of model to create ('causalimpact' or 'pymc')
            data: DataFrame containing time series data
            pre_period: List of two dates defining pre-intervention period [start, end]
            post_period: List of two dates defining post-intervention period [start, end]
            target_col: Column name of the target variable
            date_col: Column name of the date variable
            covariates: List of column names to use as covariates/controls
            model_args: Additional model-specific arguments
            
        Returns:
            A causal inference model instance
            
        Raises:
            ValueError: If the model type is not supported
        """
        model_type = model_type.lower()
        
        if model_type == "causalimpact":
            return CausalImpactModel(
                data=data,
                pre_period=pre_period,
                post_period=post_period,
                target_col=target_col,
                date_col=date_col,
                covariates=covariates,
                model_args=model_args
            )
        elif model_type == "pymc":
            return PyMCCausalModel(
                data=data,
                pre_period=pre_period,
                post_period=post_period,
                target_col=target_col,
                date_col=date_col,
                covariates=covariates,
                model_args=model_args
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}. "
                            f"Supported types are 'causalimpact' and 'pymc'.") 