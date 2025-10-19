import pandas as pd
import numpy as np

# Sample DataFrame with a column that might be inferred as 'object'
data = {
    "col1": [1, 2, np.nan, 4, 5],
    "col2": ["a", "b", "c", "d", "e"],  # This column will remain 'object'
    "col3": [10.0, np.nan, 30.0, 40.0, np.nan],
}
obs = pd.DataFrame(data)

# Infer optimal dtypes for object columns

# Now, perform the interpolation
# obs.infer_objects(copy=False)
# print(obs.dtypes)

numeric_cols = obs.select_dtypes(include=["number"]).columns
obs[numeric_cols] = obs[numeric_cols].interpolate(
    method="linear", limit_direction="both"
)
# obs.interpolate(method="linear", limit_direction="both", inplace=True)
print(obs)
