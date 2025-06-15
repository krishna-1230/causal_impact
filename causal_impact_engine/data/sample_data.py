"""
Sample datasets for demonstration.
"""
import os
from typing import Dict, Optional

import pandas as pd

from causal_impact_engine.data.data_generator import DataGenerator


class SampleData:
    """Sample datasets for demonstration."""
    
    @staticmethod
    def get_marketing_campaign_data(
        save_to_csv: bool = False,
        output_dir: Optional[str] = None,
        random_seed: Optional[int] = 42,
    ) -> pd.DataFrame:
        """
        Get sample marketing campaign data.
        
        Args:
            save_to_csv: Whether to save the data to CSV
            output_dir: Directory to save the CSV file (if save_to_csv is True)
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with sample marketing campaign data
        """
        # Generate sample data
        data = DataGenerator.generate_marketing_campaign_data(
            start_date="2022-01-01",
            end_date="2022-06-30",
            campaign_date="2022-04-01",
            campaign_effect=0.15,
            baseline_sales=1000,
            weekly_seasonality=True,
            monthly_trend=0.05,
            random_seed=random_seed,
        )
        
        # Save to CSV if requested
        if save_to_csv:
            if output_dir is None:
                output_dir = os.path.dirname(os.path.abspath(__file__))
                
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, "marketing_campaign_data.csv")
            data.to_csv(file_path, index=False)
            print(f"Saved sample marketing campaign data to {file_path}")
            
        return data
    
    @staticmethod
    def get_synthetic_data(
        effect_size: float = 0.2,
        save_to_csv: bool = False,
        output_dir: Optional[str] = None,
        random_seed: Optional[int] = 42,
    ) -> pd.DataFrame:
        """
        Get synthetic data with a causal effect.
        
        Args:
            effect_size: Size of causal effect as proportion of pre-intervention mean
            save_to_csv: Whether to save the data to CSV
            output_dir: Directory to save the CSV file (if save_to_csv is True)
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with synthetic data
        """
        # Generate synthetic data
        data = DataGenerator.generate_synthetic_data(
            pre_period_days=90,
            post_period_days=30,
            intervention_date="2023-01-01",
            effect_size=effect_size,
            trend=0.01,
            seasonality=True,
            noise_level=0.1,
            covariates_count=2,
            random_seed=random_seed,
        )
        
        # Save to CSV if requested
        if save_to_csv:
            if output_dir is None:
                output_dir = os.path.dirname(os.path.abspath(__file__))
                
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, "synthetic_data.csv")
            data.to_csv(file_path, index=False)
            print(f"Saved synthetic data to {file_path}")
            
        return data
    
    @staticmethod
    def get_ecommerce_data(
        save_to_csv: bool = False,
        output_dir: Optional[str] = None,
        random_seed: Optional[int] = 42,
    ) -> pd.DataFrame:
        """
        Get sample e-commerce data with a website redesign intervention.
        
        Args:
            save_to_csv: Whether to save the data to CSV
            output_dir: Directory to save the CSV file (if save_to_csv is True)
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with sample e-commerce data
        """
        # Set random seed
        if random_seed is not None:
            import numpy as np
            np.random.seed(random_seed)
            
        # Generate date range
        date_range = pd.date_range(start="2022-01-01", end="2022-06-30")
        n_days = len(date_range)
        
        # Generate website redesign indicator (launched on April 1, 2022)
        redesign_date = pd.to_datetime("2022-04-01")
        redesign = np.zeros(n_days)
        redesign[date_range >= redesign_date] = 1
        
        # Generate baseline metrics
        baseline_visitors = 5000
        baseline_conversion_rate = 0.02
        baseline_aov = 50  # Average order value
        
        # Generate daily visitors with trend and seasonality
        trend = 0.001 * np.arange(n_days)  # Small upward trend
        day_of_week = np.array([d.dayofweek for d in date_range])
        
        # Weekend effect on visitors
        weekend_effect = np.zeros(n_days)
        weekend_effect[day_of_week >= 5] = 0.3  # 30% more visitors on weekends
        
        # Generate visitors
        visitors = baseline_visitors * (1 + trend) * (1 + weekend_effect)
        visitors = visitors * np.random.normal(1, 0.1, n_days)  # Add noise
        
        # Website redesign effect on visitors (10% increase)
        visitors = visitors * (1 + 0.1 * redesign)
        
        # Generate conversion rate with seasonality
        conversion_rate = baseline_conversion_rate * np.ones(n_days)
        
        # Day of week effect on conversion rate
        dow_effect = np.zeros(n_days)
        dow_effect[day_of_week == 0] = -0.1  # Monday: -10%
        dow_effect[day_of_week == 1] = -0.05  # Tuesday: -5%
        dow_effect[day_of_week == 2] = 0.0  # Wednesday: 0%
        dow_effect[day_of_week == 3] = 0.05  # Thursday: +5%
        dow_effect[day_of_week == 4] = 0.1  # Friday: +10%
        dow_effect[day_of_week == 5] = 0.15  # Saturday: +15%
        dow_effect[day_of_week == 6] = 0.05  # Sunday: +5%
        
        conversion_rate = conversion_rate * (1 + dow_effect)
        
        # Website redesign effect on conversion rate (20% increase)
        conversion_rate = conversion_rate * (1 + 0.2 * redesign)
        
        # Add noise to conversion rate
        conversion_rate = conversion_rate * np.random.normal(1, 0.05, n_days)
        
        # Generate average order value with small random fluctuations
        aov = baseline_aov * np.ones(n_days)
        aov = aov * np.random.normal(1, 0.03, n_days)
        
        # Website redesign effect on AOV (5% increase)
        aov = aov * (1 + 0.05 * redesign)
        
        # Calculate orders and revenue
        orders = visitors * conversion_rate
        revenue = orders * aov
        
        # Create DataFrame
        data = pd.DataFrame({
            "date": date_range,
            "visitors": visitors.astype(int),
            "conversion_rate": conversion_rate,
            "orders": orders.astype(int),
            "aov": aov.round(2),
            "revenue": revenue.round(2),
            "redesign": redesign.astype(int)
        })
        
        # Save to CSV if requested
        if save_to_csv:
            if output_dir is None:
                output_dir = os.path.dirname(os.path.abspath(__file__))
                
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, "ecommerce_data.csv")
            data.to_csv(file_path, index=False)
            print(f"Saved e-commerce data to {file_path}")
            
        return data 