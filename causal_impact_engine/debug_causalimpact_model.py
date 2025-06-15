"""
Debug script to test our CausalImpact model with simple data
"""
import pandas as pd
import numpy as np
from causal_impact_engine.models.causal_impact_model import CausalImpactModel
from causalimpact import CausalImpact

print("Testing direct CausalImpact package...")
# Create simple test data
dates = pd.date_range(start='2022-01-01', periods=100)
data = pd.DataFrame({
    'date': dates,
    'y': np.random.normal(10, 1, 100),
    'x1': np.random.normal(5, 1, 100),
    'x2': np.random.normal(8, 1, 100)
})

# Add intervention effect
data.loc[data['date'] >= '2022-03-01', 'y'] += 2

# Set date as index for direct CausalImpact test
data_indexed = data.copy().set_index('date')

# Define pre and post periods
pre_period = ['2022-01-01', '2022-02-28']
post_period = ['2022-03-01', '2022-04-10']

# Test the original CausalImpact package
try:
    print("Creating CausalImpact object...")
    ci = CausalImpact(
        data_indexed[['y', 'x1', 'x2']],
        pd.to_datetime(pre_period),
        pd.to_datetime(post_period)
    )
    print("CausalImpact object created successfully")
    
    print("CausalImpact attributes:", dir(ci))
    
    # Check if inferences exists
    if hasattr(ci, 'inferences'):
        print("inferences attribute exists")
        if ci.inferences is not None:
            print("inferences is not None")
            print("inferences shape:", ci.inferences.shape)
            print("inferences columns:", ci.inferences.columns.tolist())
        else:
            print("inferences is None")
    else:
        print("inferences attribute does not exist")
    
    # Try to access summary
    print("Trying to access summary()...")
    summary = ci.summary()
    print("Summary:", summary[:500])  # Print first 500 chars
    
except Exception as e:
    print(f"Error with direct CausalImpact: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50 + "\n")

print("Now testing our CausalImpactModel...")
# Create and run our model
try:
    print("Creating CausalImpactModel object...")
    model = CausalImpactModel(
        data=data,
        pre_period=pre_period,
        post_period=post_period,
        target_col='y',
        date_col='date',
        covariates=['x1', 'x2']
    )
    print("CausalImpactModel object created successfully")
    
    print("Running inference...")
    model.run_inference()
    print("Model ran successfully!")
    
    print("\nResults:")
    for key, value in model.results.items():
        if key != 'inferences' and key != 'report' and key != 'model_summary':
            print(f"{key}: {value}")
    
    if model.results['inferences'] is not None:
        print("\nInferences shape:", model.results['inferences'].shape)
        print("Inferences columns:", model.results['inferences'].columns.tolist())
    else:
        print("\nInferences is None")
    
except Exception as e:
    print(f"Error running model: {e}")
    import traceback
    traceback.print_exc() 