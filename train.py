import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading Datasets..........")
data = fetch_california_housing()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"Loaded data: {X.shape}")

#================================================================================
# TRAIN AND TEST SPLIT
#================================================================================

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#================================================================================
# TRAINING A MODEL
#================================================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
) 
model.fit(X_train, y_train)

#================================================================================
# MAKE PREDICTIONS 
#================================================================================

y_pred = model.predict(X_test)

#================================================================================
# CALCULATE METRICS
#================================================================================

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

print("=" * 60)
print("METRICS")
print("=" * 60)
print(f"R² Score: {r2 * 100000:,.0f}")
print(f"MAE: {mae * 100000:,.0f}")


#=============================================================================
# JOBLIB
#=============================================================================
joblib.dump(model, "House_Model.joblib")
joblib.dump(list(X.columns),  "House_Features.joblib")