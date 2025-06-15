"""
Tests for the CausalImpact model.
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add the parent directory to the path to import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from causal_impact_engine.data.sample_data import SampleData
from causal_impact_engine.models.causal_impact_model import CausalImpactModel


class TestCausalImpactModel(unittest.TestCase):
    """Test cases for the CausalImpact model."""
    
    def setUp(self):
        """Set up test data."""
        # Generate sample data
        self.data = SampleData.get_synthetic_data(
            effect_size=0.2,
            random_seed=42
        )
        
        # Define pre and post periods
        self.pre_period = ["2022-10-03", "2022-12-31"]
        self.post_period = ["2023-01-01", "2023-01-30"]
        
    def test_model_initialization(self):
        """Test model initialization."""
        model = CausalImpactModel(
            data=self.data,
            pre_period=self.pre_period,
            post_period=self.post_period,
            target_col="y",
            date_col="date",
            covariates=["x1", "x2"]
        )
        
        self.assertIsNotNone(model)
        self.assertEqual(model.target_col, "y")
        self.assertEqual(model.date_col, "date")
        self.assertEqual(model.covariates, ["x1", "x2"])
        
    def test_model_inference(self):
        """Test model inference."""
        model = CausalImpactModel(
            data=self.data,
            pre_period=self.pre_period,
            post_period=self.post_period,
            target_col="y",
            date_col="date",
            covariates=["x1", "x2"]
        )
        
        # Run inference
        model.run_inference()
        
        # Check results
        self.assertIsNotNone(model.results)
        self.assertIn("cumulative_effect", model.results)
        self.assertIn("relative_effect", model.results)
        self.assertIn("posterior_probability", model.results)
        self.assertIn("p_value", model.results)
        
    def test_model_prediction(self):
        """Test model prediction."""
        model = CausalImpactModel(
            data=self.data,
            pre_period=self.pre_period,
            post_period=self.post_period,
            target_col="y",
            date_col="date",
            covariates=["x1", "x2"]
        )
        
        # Run inference
        model.run_inference()
        
        # Get predictions
        predictions = model.predict()
        
        # Check predictions
        self.assertIsNotNone(predictions)
        self.assertIn("prediction", predictions.columns)
        self.assertIn("prediction_lower", predictions.columns)
        self.assertIn("prediction_upper", predictions.columns)
        
    def test_model_summary(self):
        """Test model summary."""
        model = CausalImpactModel(
            data=self.data,
            pre_period=self.pre_period,
            post_period=self.post_period,
            target_col="y",
            date_col="date",
            covariates=["x1", "x2"]
        )
        
        # Run inference
        model.run_inference()
        
        # Get summary
        summary = model.get_summary()
        
        # Check summary
        self.assertIsNotNone(summary)
        self.assertIn("average_effect", summary)
        self.assertIn("cumulative_effect", summary)
        self.assertIn("relative_effect", summary)
        self.assertIn("p_value", summary)
        self.assertIn("is_significant", summary)
        
    def test_positive_effect(self):
        """Test positive effect detection."""
        # Generate data with positive effect
        data = SampleData.get_synthetic_data(
            effect_size=0.2,
            random_seed=42
        )
        
        model = CausalImpactModel(
            data=data,
            pre_period=self.pre_period,
            post_period=self.post_period,
            target_col="y",
            date_col="date",
            covariates=["x1", "x2"]
        )
        
        # Run inference
        model.run_inference()
        
        # Check effect
        summary = model.get_summary()
        self.assertGreater(summary["relative_effect"], 0)
        
    def test_negative_effect(self):
        """Test negative effect detection."""
        # Generate data with negative effect
        data = SampleData.get_synthetic_data(
            effect_size=-0.2,
            random_seed=42
        )
        
        model = CausalImpactModel(
            data=data,
            pre_period=self.pre_period,
            post_period=self.post_period,
            target_col="y",
            date_col="date",
            covariates=["x1", "x2"]
        )
        
        # Run inference
        model.run_inference()
        
        # Check effect
        summary = model.get_summary()
        self.assertLess(summary["relative_effect"], 0)


if __name__ == "__main__":
    unittest.main() 