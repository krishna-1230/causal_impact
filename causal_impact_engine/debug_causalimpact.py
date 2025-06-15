"""
Debug script to check CausalImpact output structure
"""
import pandas as pd
from causalimpact import CausalImpact

# Create simple test data
data = pd.DataFrame({
    'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'x': [1, 2, 3, 4, 5, 6, 5, 4, 3, 2]
})
data.index = pd.date_range(start='2022-01-01', periods=10)

# Define pre and post periods
pre_period = [0, 4]  # First 5 points
post_period = [5, 9]  # Last 5 points

# Run CausalImpact
ci = CausalImpact(data, pre_period, post_period)

# Print info about the output
print("CausalImpact object attributes:")
print(dir(ci))

# Check if inferences exists
if hasattr(ci, 'inferences') and ci.inferences is not None:
    print("\nAvailable columns in inferences DataFrame:")
    print(ci.inferences.columns.tolist())
    
    print("\nSample of inferences DataFrame:")
    print(ci.inferences.head())
else:
    print("\ninferences attribute is None or doesn't exist")

# Check if summary_data exists
try:
    print("\nsummary_data attribute exists:")
    print(ci.summary_data)
except AttributeError:
    print("\nsummary_data attribute does not exist")

# Print the summary
print("\nSummary output:")
print(ci.summary())

# Print the data
print("\nOriginal data:")
print(data.head())

# Try to access specific attributes
print("\nTrying to access specific attributes:")
for attr in ['data', 'params', 'model', 'series', 'summary_data']:
    try:
        value = getattr(ci, attr)
        print(f"{attr}: {type(value)}")
        if hasattr(value, 'shape'):
            print(f"{attr} shape: {value.shape}")
    except AttributeError:
        print(f"{attr} does not exist") 