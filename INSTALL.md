# Installation Guide for Causal Impact Engine

This guide provides detailed installation instructions for setting up the Causal Impact Engine on different operating systems.

## Prerequisites

- **Python 3.8+**
- **Anaconda or Miniconda** (strongly recommended, especially for Windows users)

## Windows Installation

1. **Install Anaconda or Miniconda**
   - Download from [https://www.anaconda.com/download/](https://www.anaconda.com/download/)
   - Follow the installation instructions, ensuring you add Anaconda to your PATH

2. **Clone the Repository**
   ```
   git clone https://github.com/yourusername/causal-impact-engine.git
   cd causal-impact-engine
   ```

3. **Create and Activate the Conda Environment**
   ```
   conda env create -f environment.yml
   conda activate causal_impact
   ```

4. **Run the Application**
   ```
   run_causal_impact.bat
   ```
   
   Alternatively, you can run it manually:
   ```
   cd causal_impact_engine
   streamlit run frontend/app.py
   ```

## macOS/Linux Installation

1. **Install Anaconda or Miniconda**
   - Download from [https://www.anaconda.com/download/](https://www.anaconda.com/download/)
   - Follow the installation instructions for your OS

2. **Clone the Repository**
   ```
   git clone https://github.com/yourusername/causal-impact-engine.git
   cd causal-impact-engine
   ```

3. **Create and Activate the Conda Environment**
   ```
   conda env create -f environment.yml
   conda activate causal_impact
   ```

4. **Run the Application**
   ```
   cd causal_impact_engine
   streamlit run frontend/app.py
   ```

## Troubleshooting

### PyMC Installation Issues on Windows

If you encounter issues with PyMC installation on Windows, ensure you've installed the required toolchain:

```
conda install -c conda-forge m2w64-toolchain
```

### Missing Dependencies

If you see import errors when running the application, ensure all dependencies are installed:

```
pip install -r causal_impact_engine/requirements.txt
```

### Environment Activation Problems

If conda environment activation fails, try using the full path to the activate script:

- **Windows**: `%USERPROFILE%\anaconda3\Scripts\activate.bat causal_impact`
- **macOS/Linux**: `source ~/anaconda3/bin/activate causal_impact`

## Development Installation

For development purposes, install additional tools:

```
pip install -e ".[dev]"
```

This installs the package in development mode with additional development dependencies like pytest, black, etc. 