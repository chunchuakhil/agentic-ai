
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

actual = [30, 50, 70, 80]

model_a = [32, 48, 69, 78]
model_b = [35, 44, 75, 70]
# their is no model here, just comparing those metrics.
for name, predicted in [("Model A", model_a), ("Model B", model_b)]:
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(actual, predicted))
    print("MSE:", mean_squared_error(actual, predicted))
    print("R²:", round(r2_score(actual, predicted), 2))
