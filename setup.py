"""
Setup script for the causal_impact_engine package.
"""
from setuptools import find_packages, setup

try:
    with open("causal_impact_engine/README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = """
    Causal Impact Engine: Marketing Attribution Simulator.
    
    A sophisticated tool for analyzing the causal impact of marketing campaigns
    on sales or other business metrics using Bayesian structural time series models.
    """

setup(
    name="causal_impact_engine",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A sophisticated tool for analyzing the causal impact of marketing campaigns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/causal-impact-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
        "pymc>=5.0.0",
        "arviz>=0.12.0",
        "causalimpact>=0.3.0",
        "streamlit>=1.10.0",
        "plotly>=5.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
    },
) 