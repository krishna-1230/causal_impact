# Causal Impact Engine

A sophisticated tool for analyzing the causal impact of marketing campaigns and interventions on business metrics using Bayesian structural time series models.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The Causal Impact Engine provides a robust framework for determining the true causal effect of marketing campaigns, website redesigns, pricing changes, or any other business intervention. Using Bayesian structural time series models, it goes beyond correlation to establish causation by:

1. Building a counterfactual model of what would have happened without the intervention
2. Comparing observed data against this counterfactual prediction
3. Quantifying the causal effect with statistical confidence intervals

## Key Features

- **Robust Causal Inference**: Determine if interventions truly caused changes in business metrics
- **Multiple Modeling Approaches**: Leverages both PyMC (Bayesian modeling) and Google's CausalImpact package
- **Interactive Visualization**: Explore results through an intuitive Streamlit interface
- **Comprehensive Analysis**: Provides counterfactual predictions, point estimates, and credible intervals
- **Flexible Input**: Works with various time-series data formats and covariates

## Installation

This project requires **Anaconda/Miniconda** for environment management, particularly due to dependencies needed for PyMC on Windows.

```bash
# Clone the repository
git https://github.com/krishna-1230/causal_impact.git
cd causal-impact-engine

# Create conda environment
conda env create -f environment.yml
conda activate causal_impact

# Install Windows-specific dependencies (required for PyMC)
conda install -c conda-forge m2w64-toolchain

# Install Python dependencies
pip install -r causal_impact_engine/requirements.txt
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Quick Start

### Running the Web Application

The easiest way to use the Causal Impact Engine is through its Streamlit interface:

#### Windows
```bash
# Run the provided batch file
run_causal_impact.bat
```

#### Linux/macOS
```bash
# Make the script executable (first time only)
chmod +x run_causal_impact.sh

# Run the script
./run_causal_impact.sh
```

Alternatively, you can run it manually:

```bash
# Activate the conda environment
conda activate causal_impact

# Run the Streamlit app
cd causal_impact_engine
streamlit run frontend/app.py
```

### Using the API Programmatically

```python
from causal_impact_engine.models.causal_impact_model import CausalImpactModel
import pandas as pd

# Load your time series data
data = pd.read_csv('your_data.csv')

# Define pre and post-intervention periods
pre_period = ['2022-01-01', '2022-02-28']
post_period = ['2022-03-01', '2022-04-30']

# Initialize and run the model
model = CausalImpactModel(
    data=data, 
    pre_period=pre_period, 
    post_period=post_period,
    target_col='sales',
    date_col='date',
    covariates=['web_traffic', 'ad_spend']
)

# Run inference
impact = model.run_inference()

# Get results
summary = impact.get_summary()
print(f"Relative effect: {summary['relative_effect'] * 100:.2f}%")
print(f"Probability of causal effect: {summary['posterior_probability']:.2f}")

# Visualize results
fig = model.plot_results()
fig.savefig('causal_impact_results.png')
```

## Project Structure

```
causal_impact_engine/
├── data/               # Sample datasets and data loaders
│   ├── data_generator.py  # Synthetic data generation
│   ├── data_loader.py     # Data loading utilities
│   └── sample_data.py     # Pre-built sample datasets
├── models/             # Causal inference models
│   ├── base_model.py      # Abstract base class
│   ├── causal_impact_model.py  # Google CausalImpact implementation
│   ├── model_factory.py   # Factory pattern for model creation
│   └── pymc_model.py      # PyMC-based implementation
├── utils/              # Helper functions and utilities
│   ├── metrics.py         # Evaluation metrics
│   ├── reporting.py       # Report generation
│   └── visualization.py   # Plotting functions
├── frontend/           # Streamlit web application
│   └── app.py            # Main Streamlit interface
├── tests/              # Unit and integration tests
├── notebooks/          # Jupyter notebooks for exploration
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Examples

The `examples/` directory contains ready-to-use examples:

- **Marketing Campaign Analysis**: Measure the impact of a marketing campaign on sales
- **Website Redesign Evaluation**: Assess how a website redesign affected conversion rates

## Requirements

- Python 3.8+
- Anaconda/Miniconda
- Dependencies listed in `requirements.txt`

## License

MIT

## Citation

If you use this tool in your research, please cite:
```
@software{causal_impact_engine,
  author = {Your Name},
  title = {Causal Impact Engine: Marketing Attribution Simulator},
  year = {2023},
  url = {https://github.com/yourusername/causal-impact-engine}
}
```
