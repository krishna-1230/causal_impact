"""
Visualization utilities for causal impact analysis.
"""
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


class CausalImpactVisualizer:
    """Visualization utilities for causal impact analysis."""
    
    @staticmethod
    def plot_time_series(
        data: pd.DataFrame,
        date_col: str = "date",
        target_col: str = "y",
        intervention_date: Optional[str] = None,
        title: str = "Time Series with Intervention",
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot a time series with an optional intervention line.
        
        Args:
            data: DataFrame with time series data
            date_col: Name of the date column
            target_col: Name of the target column
            intervention_date: Date of intervention (if None, no intervention line)
            title: Plot title
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot time series
        ax.plot(data[date_col], data[target_col])
        
        # Add intervention line if provided
        if intervention_date is not None:
            intervention_date_dt = pd.to_datetime(intervention_date)
            ax.axvline(intervention_date_dt, color="r", linestyle="--", label="Intervention")
            
            # Add pre/post labels
            y_max = data[target_col].max()
            ax.text(
                pd.to_datetime(data[date_col].iloc[0]) + 
                0.25 * (intervention_date_dt - pd.to_datetime(data[date_col].iloc[0])),
                y_max * 0.9,
                "Pre-Intervention",
                ha="center"
            )
            ax.text(
                intervention_date_dt + 
                0.5 * (pd.to_datetime(data[date_col].iloc[-1]) - intervention_date_dt),
                y_max * 0.9,
                "Post-Intervention",
                ha="center"
            )
        
        # Set labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel(target_col)
        ax.set_title(title)
        
        # Add legend if intervention line was added
        if intervention_date is not None:
            ax.legend()
            
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_actual_vs_counterfactual(
        actual: pd.DataFrame,
        counterfactual: pd.DataFrame,
        date_col: str = "date",
        actual_col: str = "y",
        counterfactual_col: str = "prediction",
        lower_col: str = "prediction_lower",
        upper_col: str = "prediction_upper",
        intervention_date: Optional[str] = None,
        title: str = "Actual vs Counterfactual",
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot actual values against counterfactual predictions.
        
        Args:
            actual: DataFrame with actual values
            counterfactual: DataFrame with counterfactual predictions
            date_col: Name of the date column
            actual_col: Name of the actual values column
            counterfactual_col: Name of the counterfactual predictions column
            lower_col: Name of the lower bound column
            upper_col: Name of the upper bound column
            intervention_date: Date of intervention
            title: Plot title
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual values
        ax.plot(actual[date_col], actual[actual_col], label="Actual")
        
        # Plot counterfactual predictions
        ax.plot(counterfactual[date_col], counterfactual[counterfactual_col], 
                linestyle="--", label="Counterfactual")
        
        # Add confidence interval
        ax.fill_between(
            counterfactual[date_col],
            counterfactual[lower_col],
            counterfactual[upper_col],
            alpha=0.2,
            color="g",
            label="95% Credible Interval"
        )
        
        # Add intervention line if provided
        if intervention_date is not None:
            intervention_date_dt = pd.to_datetime(intervention_date)
            ax.axvline(intervention_date_dt, color="r", linestyle="--", label="Intervention")
        
        # Set labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_cumulative_effect(
        data: pd.DataFrame,
        date_col: str = "date",
        effect_col: str = "effect",
        intervention_date: Optional[str] = None,
        title: str = "Cumulative Effect",
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot the cumulative effect of an intervention.
        
        Args:
            data: DataFrame with effect data
            date_col: Name of the date column
            effect_col: Name of the effect column
            intervention_date: Date of intervention
            title: Plot title
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate cumulative effect
        data = data.copy()
        data["cumulative_effect"] = data[effect_col].cumsum()
        
        # Plot cumulative effect
        ax.plot(data[date_col], data["cumulative_effect"])
        
        # Add zero line
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.2)
        
        # Add intervention line if provided
        if intervention_date is not None:
            intervention_date_dt = pd.to_datetime(intervention_date)
            ax.axvline(intervention_date_dt, color="r", linestyle="--", label="Intervention")
        
        # Set labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Effect")
        ax.set_title(title)
        
        # Add legend if intervention line was added
        if intervention_date is not None:
            ax.legend()
            
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_posterior_distribution(
        samples: np.ndarray,
        effect_mean: float,
        effect_lower: float,
        effect_upper: float,
        title: str = "Posterior Distribution of Effect",
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Plot the posterior distribution of the causal effect.
        
        Args:
            samples: Array of posterior samples
            effect_mean: Mean effect
            effect_lower: Lower bound of credible interval
            effect_upper: Upper bound of credible interval
            title: Plot title
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot posterior distribution
        sns.histplot(samples, kde=True, ax=ax)
        
        # Add vertical lines for mean and CI
        ax.axvline(effect_mean, color="r", linestyle="-", label="Mean")
        ax.axvline(effect_lower, color="r", linestyle="--", label="95% CI")
        ax.axvline(effect_upper, color="r", linestyle="--")
        
        # Add zero line
        ax.axvline(0, color="k", linestyle="-", alpha=0.2, label="No Effect")
        
        # Set labels and title
        ax.set_xlabel("Effect Size")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_interactive_plot(
        data: pd.DataFrame,
        date_col: str = "date",
        target_col: str = "y",
        counterfactual_col: str = "prediction",
        lower_col: str = "prediction_lower",
        upper_col: str = "prediction_upper",
        effect_col: str = "effect",
        intervention_date: Optional[str] = None,
        title: str = "Causal Impact Analysis",
    ) -> go.Figure:
        """
        Create an interactive Plotly figure for causal impact analysis.
        
        Args:
            data: DataFrame with actual and counterfactual data
            date_col: Name of the date column
            target_col: Name of the target column
            counterfactual_col: Name of the counterfactual column
            lower_col: Name of the lower bound column
            upper_col: Name of the upper bound column
            effect_col: Name of the effect column
            intervention_date: Date of intervention
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=3, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Original vs Counterfactual", "Pointwise Effect", "Cumulative Effect")
        )
        
        # Calculate cumulative effect
        data = data.copy()
        data["cumulative_effect"] = data[effect_col].cumsum()
        
        # Get the dates - handle case when date_col is the index
        if date_col == data.index.name or data.index.name is None and isinstance(data.index, pd.DatetimeIndex):
            dates = data.index
        else:
            dates = data[date_col]
        
        # Add actual data
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=data[target_col],
                mode="lines",
                name="Actual",
                line=dict(color="blue")
            ),
            row=1, col=1
        )
        
        # Add counterfactual
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=data[counterfactual_col],
                mode="lines",
                name="Counterfactual",
                line=dict(color="green", dash="dash")
            ),
            row=1, col=1
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=dates.tolist() + dates.tolist()[::-1],
                y=data[upper_col].tolist() + data[lower_col].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(0, 255, 0, 0.1)",
                line=dict(color="rgba(0, 0, 0, 0)"),
                name="95% Credible Interval"
            ),
            row=1, col=1
        )
        
        # Add pointwise effect
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=data[effect_col],
                mode="lines",
                name="Pointwise Effect",
                line=dict(color="red")
            ),
            row=2, col=1
        )
        
        # Add zero line for pointwise effect
        fig.add_trace(
            go.Scatter(
                x=[dates.min(), dates.max()],
                y=[0, 0],
                mode="lines",
                name="Zero Line",
                line=dict(color="black", dash="dash", width=1),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add cumulative effect
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=data["cumulative_effect"],
                mode="lines",
                name="Cumulative Effect",
                line=dict(color="purple")
            ),
            row=3, col=1
        )
        
        # Add zero line for cumulative effect
        fig.add_trace(
            go.Scatter(
                x=[dates.min(), dates.max()],
                y=[0, 0],
                mode="lines",
                name="Zero Line",
                line=dict(color="black", dash="dash", width=1),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Add intervention line if provided
        if intervention_date is not None:
            intervention_date_dt = pd.to_datetime(intervention_date)
            
            for row in range(1, 4):
                fig.add_vline(
                    x=intervention_date_dt,
                    line_width=2,
                    line_dash="dash",
                    line_color="red",
                    row=row,
                    col=1
                )
                
            # Add annotation
            fig.add_annotation(
                x=intervention_date_dt,
                y=data[target_col].max(),
                text="Intervention",
                showarrow=True,
                arrowhead=1,
                row=1,
                col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Effect", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Effect", row=3, col=1)
        
        return fig 