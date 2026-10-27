
# SkillMatch — Internship Readiness Analyzer

SkillMatch is a machine-learning web application that explores internship readiness using a candidate's technical skills, project experience, coding practice, computer science fundamentals, and communication skills.

## Features

- Candidate assessment through a web form
- Predictions using Logistic Regression and Random Forest
- Comparison of model accuracy, precision, and recall
- Profile-based preparation suggestions
- Interactive results dashboard built with HTML, CSS, and JavaScript

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn
- **Data Processing:** pandas
- **Model Persistence:** joblib

## Project Structure

```text
skillmatch/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── features.pkl
├── logistic_model.pkl
├── random_forest_model.pkl
├── metrics.pkl
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```

## Run Locally

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   python app.py
   ```

5. Open `http://127.0.0.1:5000` in your browser.

The saved model files are included, so you do not need to retrain the models to run the application. To regenerate them, run `python train_model.py`.

## How It Works

The application accepts five candidate inputs and sends them to a Flask endpoint. Two trained classification models independently predict either `Ready` or `Needs Preparation`. The results page displays both predictions and evaluation metrics calculated on a held-out test set.

Preparation suggestions are generated using simple input-based rules; they are not produced by the machine-learning models.

## Dataset and Limitations

This is an educational prototype trained on a small synthetic dataset. The target labels are generated using predefined rules, so the evaluation scores reflect performance on those generated labels rather than validated real-world internship outcomes.

The predictions should not be interpreted as guarantees of internship selection or as scientifically validated readiness assessments.

## Future Improvements

- Train and evaluate using a larger, representative dataset with validated labels.
- Use cross-validation and evaluate generalisation on independent data.
- Improve the transparency and validation of readiness criteria.
- Add more detailed, explainable feedback for candidates.
