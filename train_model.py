import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# 1. Create sample candidate data
data = {
    "skills_score": [80, 30, 65, 45, 90, 25, 70, 35, 85, 50,
                     75, 20, 60, 40, 95, 55, 68, 32, 88, 48,
                     78, 28, 62, 42, 92, 38, 73, 33, 82, 52],
    "projects": [3, 0, 2, 1, 4, 0, 3, 1, 4, 1,
                 3, 0, 2, 1, 5, 2, 2, 0, 4, 1,
                 3, 0, 2, 1, 5, 1, 3, 0, 4, 2],
    "coding_hours": [8, 1, 5, 2, 10, 1, 7, 2, 9, 3,
                     6, 1, 4, 2, 10, 4, 6, 1, 9, 3,
                     7, 1, 5, 2, 10, 2, 6, 1, 8, 4],
    "cs_score": [85, 35, 70, 40, 95, 25, 75, 30, 90, 50,
                 80, 20, 65, 45, 98, 55, 72, 28, 92, 48,
                 78, 25, 68, 42, 96, 38, 74, 32, 88, 52],
    "communication": [ 75, 40, 65, 50, 90, 35, 70, 45, 85, 55,
                       80, 30, 60, 48, 95, 58, 68, 38, 88, 52,
                       72, 42, 78, 46, 92, 56, 66, 34, 84, 60
]
}
for column, values in data.items():
    print(column, len(values))
df = pd.DataFrame(data)

# 2. Create the target: 1 = Ready, 0 = Needs Preparation
df["ready"] = (
    (df["skills_score"] >= 60) &
    (df["projects"] >= 2) &
    (df["cs_score"] >= 60)
).astype(int)

# 3. Separate inputs (X) and target (y)
X = df.drop(columns=["ready"])
y = df["ready"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Create two ML models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}
model_metrics = {}

# 6. Train, predict, and evaluate both models
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)

    model_metrics[name] = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall
    }
    print(f"\n{name}")
    print("-" * 30)
    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")

# 7. Example candidate prediction using both models
candidate = pd.DataFrame([{
    "skills_score": 75,
    "projects": 2,
    "coding_hours": 5,
    "cs_score": 70,
    "communication": 65
}])

print("\nExample Candidate Prediction")
for name, model in models.items():
    result = model.predict(candidate)[0]
    label = "Ready" if result == 1 else "Needs Preparation"
    print(f"{name}: {label}")

# Save both trained models for the Flask application
joblib.dump(models["Logistic Regression"], "logistic_model.pkl")
joblib.dump(models["Random Forest"], "random_forest_model.pkl")

# Save the feature names and order
joblib.dump(list(X.columns), "features.pkl")
joblib.dump(model_metrics, "metrics.pkl")
print("\nModels saved successfully!")