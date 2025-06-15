"""
Data generator for creating synthetic time series data with causal effects.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


class DataGenerator:
    """Generate synthetic time series data with causal effects."""
    
    @staticmethod
    def generate_synthetic_data(
        pre_period_days: int = 90,
        post_period_days: int = 30,
        intervention_date: str = "2023-01-01",
        effect_size: float = 0.2,
        trend: float = 0.01,
        seasonality: bool = True,
        noise_level: float = 0.1,
        covariates_count: int = 2,
        covariate_effects: Optional[List[float]] = None,
        random_seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic time series data with a causal effect.
        
        Args:
            pre_period_days: Number of days in pre-intervention period
            post_period_days: Number of days in post-intervention period
            intervention_date: Date of intervention as string (YYYY-MM-DD)
            effect_size: Size of causal effect as proportion of pre-intervention mean
            trend: Daily trend coefficient
            seasonality: Whether to include weekly seasonality
            noise_level: Standard deviation of random noise
            covariates_count: Number of covariates to generate
            covariate_effects: List of coefficients for covariates
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with synthetic time series data
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Set default covariate effects if not provided
        if covariate_effects is None:
            covariate_effects = [0.5, -0.3] + [0.0] * (covariates_count - 2)
        
        # Ensure we have the right number of covariate effects
        if len(covariate_effects) < covariates_count:
            covariate_effects.extend([0.0] * (covariates_count - len(covariate_effects)))
        elif len(covariate_effects) > covariates_count:
            covariate_effects = covariate_effects[:covariates_count]
        
        # Generate dates
        total_days = pre_period_days + post_period_days
        intervention_date_dt = pd.to_datetime(intervention_date)
        start_date = intervention_date_dt - pd.Timedelta(days=pre_period_days)
        dates = [start_date + pd.Timedelta(days=i) for i in range(total_days)]
        
        # Generate time index and intervention indicator
        time_idx = np.arange(total_days)
        intervention = np.zeros(total_days)
        intervention[pre_period_days:] = 1
        
        # Generate covariates
        covariates = {}
        for i in range(covariates_count):
            # Generate with some autocorrelation
            x = np.zeros(total_days)
            x[0] = np.random.normal(0, 1)
            for t in range(1, total_days):
                x[t] = 0.8 * x[t-1] + 0.2 * np.random.normal(0, 1)
            covariates[f"x{i+1}"] = x
        
        # Calculate covariate contribution
        covariate_contribution = np.zeros(total_days)
        for i, effect in enumerate(covariate_effects):
            covariate_contribution += effect * covariates[f"x{i+1}"]
        
        # Generate trend component
        trend_component = trend * time_idx
        
        # Generate seasonal component (weekly seasonality)
        if seasonality:
            days_of_week = np.array([d.dayofweek for d in dates])
            seasonal_component = 0.1 * np.sin(2 * np.pi * days_of_week / 7)
        else:
            seasonal_component = np.zeros(total_days)
        
        # Generate baseline (without intervention effect)
        baseline = 10 + trend_component + seasonal_component + covariate_contribution
        
        # Add intervention effect
        pre_mean = np.mean(baseline[:pre_period_days])
        intervention_effect = effect_size * pre_mean * intervention
        
        # Add noise
        noise = np.random.normal(0, noise_level * pre_mean, total_days)
        
        # Generate final time series
        y = baseline + intervention_effect + noise
        
        # Create DataFrame
        data = pd.DataFrame({
            "date": dates,
            "y": y,
            "intervention": intervention,
        })
        
        # Add covariates to DataFrame
        for i in range(covariates_count):
            data[f"x{i+1}"] = covariates[f"x{i+1}"]
        
        return data
    
    @staticmethod
    def generate_marketing_campaign_data(
        start_date: str = "2022-01-01",
        end_date: str = "2022-06-30",
        campaign_date: str = "2022-04-01",
        campaign_effect: float = 0.15,
        baseline_sales: float = 1000,
        weekly_seasonality: bool = True,
        monthly_trend: float = 0.05,
        random_seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic marketing campaign data with a causal effect on sales.
        
        Args:
            start_date: Start date of the data
            end_date: End date of the data
            campaign_date: Date when the marketing campaign started
            campaign_effect: Effect size of the campaign as proportion of baseline sales
            baseline_sales: Baseline daily sales
            weekly_seasonality: Whether to include weekly seasonality
            monthly_trend: Monthly trend in sales as proportion
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with synthetic marketing campaign data
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date)
        n_days = len(date_range)
        
        # Generate campaign indicator
        campaign_date_dt = pd.to_datetime(campaign_date)
        campaign = np.zeros(n_days)
        campaign[date_range >= campaign_date_dt] = 1
        
        # Generate trend component (monthly)
        days_since_start = np.arange(n_days)
        monthly_trend_daily = monthly_trend / 30
        trend_component = baseline_sales * (1 + monthly_trend_daily * days_since_start)
        
        # Generate seasonal component (weekly)
        if weekly_seasonality:
            days_of_week = np.array([d.dayofweek for d in date_range])
            # Weekend effect
            weekend_effect = np.zeros(n_days)
            weekend_effect[days_of_week >= 5] = 0.2  # 20% increase on weekends
            
            # Day of week effect
            dow_effect = np.zeros(n_days)
            dow_effect[days_of_week == 0] = -0.1  # Monday: -10%
            dow_effect[days_of_week == 1] = -0.05  # Tuesday: -5%
            dow_effect[days_of_week == 2] = 0.0  # Wednesday: 0%
            dow_effect[days_of_week == 3] = 0.05  # Thursday: +5%
            dow_effect[days_of_week == 4] = 0.1  # Friday: +10%
            dow_effect[days_of_week == 5] = 0.2  # Saturday: +20%
            dow_effect[days_of_week == 6] = 0.15  # Sunday: +15%
            
            seasonal_component = baseline_sales * dow_effect
        else:
            seasonal_component = np.zeros(n_days)
        
        # Generate web traffic as a covariate
        web_traffic = np.zeros(n_days)
        web_traffic[0] = baseline_sales * 0.1 * np.random.normal(1, 0.1)
        for t in range(1, n_days):
            web_traffic[t] = 0.8 * web_traffic[t-1] + 0.2 * baseline_sales * 0.1 * np.random.normal(1, 0.1)
            
        # Add campaign effect to web traffic
        web_traffic_campaign_effect = 0.3  # 30% increase in web traffic due to campaign
        web_traffic = web_traffic * (1 + web_traffic_campaign_effect * campaign)
        
        # Generate advertising spend as another covariate
        ad_spend = np.zeros(n_days)
        # Base ad spend
        ad_spend = baseline_sales * 0.05 * np.ones(n_days)
        # Add campaign boost
        ad_spend = ad_spend * (1 + 2 * campaign)  # 3x ad spend during campaign
        # Add some noise
        ad_spend = ad_spend * np.random.normal(1, 0.1, n_days)
        
        # Campaign effect on sales
        campaign_effect_component = baseline_sales * campaign_effect * campaign
        
        # Web traffic effect on sales (elasticity of 0.2)
        web_traffic_effect = 0.2 * web_traffic
        
        # Ad spend effect on sales (elasticity of 0.1)
        ad_spend_effect = 0.1 * ad_spend
        
        # Generate final sales
        sales = trend_component + seasonal_component + campaign_effect_component + web_traffic_effect + ad_spend_effect
        
        # Add noise (5% of baseline)
        noise = np.random.normal(0, 0.05 * baseline_sales, n_days)
        sales = sales + noise
        
        # Create DataFrame
        data = pd.DataFrame({
            "date": date_range,
            "sales": sales,
            "web_traffic": web_traffic,
            "ad_spend": ad_spend,
            "campaign": campaign
        })
        
        return data 