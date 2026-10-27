
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load the trained ML models
logistic_model = joblib.load("logistic_model.pkl")
forest_model = joblib.load("random_forest_model.pkl")
features = joblib.load("features.pkl")
metrics = joblib.load("metrics.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert the submitted values into a table
    candidate = pd.DataFrame(
        [[
            float(data["skills_score"]),
            int(data["projects"]),
            float(data["coding_hours"]),
            float(data["cs_score"]),
            float(data["communication"])
        ]],
        columns=features
    )

    # Get predictions from both models
    logistic_result = int(logistic_model.predict(candidate)[0])
    forest_result = int(forest_model.predict(candidate)[0])

    return jsonify({
    "logistic": "Ready" if logistic_result == 1 else "Needs Preparation",
    "forest": "Ready" if forest_result == 1 else "Needs Preparation",
    "metrics": metrics
    })


if __name__ == "__main__":
    app.run(debug=True)
