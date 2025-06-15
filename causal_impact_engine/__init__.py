"""
Causal Impact Engine: Marketing Attribution Simulator.

A sophisticated tool for analyzing the causal impact of marketing campaigns
on sales or other business metrics using Bayesian structural time series models.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from causal_impact_engine.models.base_model import BaseCausalModel
from causal_impact_engine.models.causal_impact_model import CausalImpactModel
from causal_impact_engine.models.model_factory import ModelFactory
from causal_impact_engine.models.pymc_model import PyMCCausalModel

__all__ = [
    "BaseCausalModel",
    "CausalImpactModel",
    "PyMCCausalModel",
    "ModelFactory",
] 