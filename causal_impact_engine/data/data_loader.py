"""
Data loader for loading and preprocessing real-world data.
"""
import os
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


class DataLoader:
    """Load and preprocess real-world data for causal impact analysis."""
    
    @staticmethod
    def load_csv(
        file_path: str,
        date_col: str = "date",
        date_format: Optional[str] = None,
        target_col: Optional[str] = None,
        covariates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            date_col: Name of the date column
            date_format: Format of the date column (if None, infer)
            target_col: Name of the target column
            covariates: List of covariate columns to include
            
        Returns:
            DataFrame with loaded data
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Load data
        data = pd.read_csv(file_path)
        
        # Convert date column to datetime
        try:
            data[date_col] = pd.to_datetime(data[date_col], format=date_format)
        except Exception as e:
            raise ValueError(f"Could not convert {date_col} to datetime: {e}")
            
        # Sort by date
        data = data.sort_values(by=date_col)
        
        # Select columns if specified
        if target_col is not None and covariates is not None:
            cols_to_keep = [date_col, target_col] + covariates
            data = data[cols_to_keep]
        elif target_col is not None:
            cols_to_keep = [date_col, target_col]
            data = data[cols_to_keep]
            
        return data
    
    @staticmethod
    def preprocess(
        data: pd.DataFrame,
        date_col: str = "date",
        target_col: str = "y",
        covariates: Optional[List[str]] = None,
        resample_freq: Optional[str] = None,
        fill_method: str = "ffill",
        normalize: bool = False,
    ) -> pd.DataFrame:
        """
        Preprocess data for causal impact analysis.
        
        Args:
            data: DataFrame with time series data
            date_col: Name of the date column
            target_col: Name of the target column
            covariates: List of covariate columns
            resample_freq: Frequency to resample data to (e.g., 'D', 'W', 'M')
            fill_method: Method to fill missing values ('ffill', 'bfill', 'linear', 'mean')
            normalize: Whether to normalize the data
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Set date as index for resampling
        df = df.set_index(date_col)
        
        # Resample if requested
        if resample_freq is not None:
            # Define columns to aggregate
            agg_dict = {}
            agg_dict[target_col] = 'mean'
            
            if covariates is not None:
                for cov in covariates:
                    agg_dict[cov] = 'mean'
                    
            # Resample and aggregate
            df = df.resample(resample_freq).agg(agg_dict)
        
        # Handle missing values
        if fill_method == 'ffill':
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')  # In case there are NaNs at the beginning
        elif fill_method == 'bfill':
            df = df.fillna(method='bfill')
            df = df.fillna(method='ffill')  # In case there are NaNs at the end
        elif fill_method == 'linear':
            df = df.interpolate(method='linear')
        elif fill_method == 'mean':
            df = df.fillna(df.mean())
        else:
            raise ValueError(f"Unsupported fill method: {fill_method}")
            
        # Normalize if requested
        if normalize:
            # Normalize target
            df[target_col] = (df[target_col] - df[target_col].mean()) / df[target_col].std()
            
            # Normalize covariates
            if covariates is not None:
                for cov in covariates:
                    df[cov] = (df[cov] - df[cov].mean()) / df[cov].std()
        
        # Reset index to get date as a column again
        df = df.reset_index()
        
        return df
    
    @staticmethod
    def add_time_features(
        data: pd.DataFrame,
        date_col: str = "date",
        add_day_of_week: bool = True,
        add_month: bool = True,
        add_quarter: bool = True,
        add_year: bool = True,
        add_holiday: bool = False,
        country: str = "US",
    ) -> pd.DataFrame:
        """
        Add time-based features to the data.
        
        Args:
            data: DataFrame with time series data
            date_col: Name of the date column
            add_day_of_week: Whether to add day of week feature
            add_month: Whether to add month feature
            add_quarter: Whether to add quarter feature
            add_year: Whether to add year feature
            add_holiday: Whether to add holiday indicators
            country: Country for holidays (if add_holiday is True)
            
        Returns:
            DataFrame with added time features
        """
        df = data.copy()
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Add day of week
        if add_day_of_week:
            df["day_of_week"] = df[date_col].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            
        # Add month
        if add_month:
            df["month"] = df[date_col].dt.month
            
        # Add quarter
        if add_quarter:
            df["quarter"] = df[date_col].dt.quarter
            
        # Add year
        if add_year:
            df["year"] = df[date_col].dt.year
            
        # Add holidays
        if add_holiday:
            try:
                from holidays import country_holidays
                
                # Get holidays for the specified country
                start_year = df[date_col].min().year
                end_year = df[date_col].max().year
                holidays_dict = country_holidays(country, years=range(start_year, end_year + 1))
                
                # Create holiday indicator
                df["is_holiday"] = df[date_col].isin(holidays_dict).astype(int)
                
            except ImportError:
                print("Warning: holidays package not installed. Skipping holiday features.")
                print("Install with: pip install holidays")
        
        return df 