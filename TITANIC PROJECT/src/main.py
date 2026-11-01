"""
Titanic Survival Prediction
Run: python main.py
Expected input files: dataset/train.csv and dataset/test.csv
"""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def add_features(df):
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df

def build_preprocessor():
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
    categorical_features = ["Sex", "Embarked"]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ])

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42
        )
    }

def main():
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Place train.csv and test.csv inside the dataset/ folder."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y = train_df["Survived"]
    X = add_features(train_df.drop(columns=["Survived"]))
    X_test = add_features(test_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    results = []
    fitted_models = {}
    for name, classifier in get_models().items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_val)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_val, predictions),
            "Precision": precision_score(y_val, predictions, zero_division=0),
            "Recall": recall_score(y_val, predictions, zero_division=0),
            "F1": f1_score(y_val, predictions, zero_division=0)
        })
        fitted_models[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    best_name = results_df.iloc[0]["Model"]
    print("Validation results:")
    print(results_df.to_string(index=False))
    print(f"\nSelected model by F1-score: {best_name}")

    final_model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", get_models()[best_name])
    ])
    final_model.fit(X, y)
    test_predictions = final_model.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_predictions
    })
    submission.to_csv(OUTPUT_DIR / "titanic_submission.csv", index=False)
    print(f"Saved {len(submission)} predictions to outputs/titanic_submission.csv")

if __name__ == "__main__":
    main()
