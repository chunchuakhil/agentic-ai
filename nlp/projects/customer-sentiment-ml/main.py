# %% Step 1: Load the dataset
import pandas as pd

df = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")
df.head()
# %%

# %% Step 2: Check dataset size
df.shape

# %% Step 3: Check column names
df.columns

# %%
# %% Step 2: Split data into training and testing sets
from sklearn.model_selection import train_test_split

X = df["Review"]  # Input: customer review
y = df["Liked"]   # Output: sentiment (1 = positive, 0 = negative)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# %%
# %% Step 3: Convert text into numbers using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)

# %% Step 4: Train Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")
# %%
# %% Step 5: Test the model
y_pred = model.predict(X_test_tfidf)

print(y_pred[:10])
# %%
# %% Step 6: Compare actual vs predicted
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(results.head(10))
# %%
# %% Step 7: Calculate accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Model accuracy:", accuracy * 100, "%")
# %%
# %% Step 8: Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
# %%
# %% Step 10: Classification Report
from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    y_pred,
    target_names=["Negative", "Positive"]
))
# %% Step 11: Predict a new customer review

new_review = ["The food was delicious and the service was excellent"]

new_review_tfidf = tfidf.transform(new_review)

prediction = model.predict(new_review_tfidf)

sentiment = "Positive" if prediction[0] == 1 else "Negative"

print("Review:", new_review[0])
print("Predicted sentiment:", sentiment)
# %%
