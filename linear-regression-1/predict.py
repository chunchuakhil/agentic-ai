
import joblib
# predict study hours to marks, based on custom training data.
# loaded my own existing model from linear_regression.py
model = joblib.load("linear_model.pkl")

print(model.predict([[7]]))
print(model.predict([[7.5]]))
