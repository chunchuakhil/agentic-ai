from sklearn.linear_model import LinearRegression
import joblib

X = [[1], [2], [3], [4], [5]]
y = [20, 40, 60, 80, 100]

model = LinearRegression()
model.fit(X, y)
# joblib.dump() — saves an object to a file.
joblib.dump(model, "linear_model.pkl")
