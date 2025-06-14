# Causal Impact Engine: Marketing Attribution Simulator

A sophisticated tool for analyzing the causal impact of marketing campaigns on sales or other business metrics using Bayesian structural time series models.

## Features

- **Robust Causal Inference**: Determine if marketing campaigns truly caused an increase in sales, beyond mere correlation
- **Multiple Modeling Approaches**: Leverages both PyMC (Bayesian modeling) and Google's CausalImpact package
- **Interactive Visualization**: Explore results through an intuitive Streamlit interface
- **Comprehensive Analysis**: Provides counterfactual predictions, point estimates, and credible intervals
- **Flexible Input**: Works with various time-series data formats

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/causal-impact-engine.git
cd causal-impact-engine

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Web Application

```bash
cd causal_impact_engine
streamlit run frontend/app.py
```

### Using the API Programmatically

```python
from causal_impact_engine.models.causal_impact_model import CausalImpactModel

# Load your time series data
data = pd.read_csv('your_data.csv')

# Define pre and post-intervention periods
pre_period = ['2022-01-01', '2022-02-28']
post_period = ['2022-03-01', '2022-04-30']

# Initialize and run the model
model = CausalImpactModel(data, pre_period, post_period)
impact = model.run_inference()

# Get results and visualize
summary = impact.get_summary()
model.plot_results()
```

## Project Structure

```
causal_impact_engine/
├── data/               # Sample datasets and data loaders
├── models/             # Causal inference models
├── utils/              # Helper functions and utilities
├── tests/              # Unit and integration tests
├── notebooks/          # Jupyter notebooks for exploration
├── frontend/           # Streamlit web application
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

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
``` "# causal_impact" 
