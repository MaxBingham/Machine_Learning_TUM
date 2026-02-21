import numpy as np
from active_learning import select_next_batch
from uncertainty import Committee
from sklearn.ensemble import RandomForestRegressor

# Create dummy data
X_pool = np.random.rand(10, 2)
y_dummy = np.random.rand(10)

# Create a simple model
base_est = RandomForestRegressor(n_estimators=5, random_state=42)
committee = Committee(base_est, n_estimators=3)
committee.fit(X_pool, y_dummy)

# Test select_next_batch
indices = select_next_batch(committee, X_pool, k=3)
print("Selected indices:", indices)