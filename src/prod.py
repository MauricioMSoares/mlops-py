import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data["PRICE"] = housing.target

print(data.head())

x = data.drop("PRICE", axis=1)
y = data["PRICE"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-Squared: {r2}")

# Save the model with joblib.dump, then load it with joblib.load
# without having to re-train it.
joblib.dump(model, "california_housing_model.pkl")
loaded_model = joblib.load("california_housing_model.pkl")

loaded_model_predictions = loaded_model.predict(x_test_scaled)

print(f"Predictions from loaded model: {loaded_model_predictions[:5]}")
