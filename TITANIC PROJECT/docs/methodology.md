# Methodology

## 1. Dataset
The project uses the Titanic passenger dataset. The labelled training file contains 891 records and 12 columns, while the test file contains 418 records and 11 columns. The target variable is `Survived`, where 0 indicates non-survival and 1 indicates survival.

## 2. Data preparation
The features used for modelling are passenger class, sex, age, number of siblings or spouses aboard, number of parents or children aboard, fare, and embarkation port. The passenger identifier, name, ticket, and cabin fields are not included in the model feature set.

Missing numerical values are replaced using the median of the relevant training feature. Missing categorical values are replaced using the most frequent category. Numerical variables are standardized, and categorical variables are converted into one-hot encoded columns. The preprocessing operations are placed inside a scikit-learn `ColumnTransformer` and `Pipeline` to ensure that transformations are fitted using training data only.

## 3. Feature engineering
Two additional features are derived. `FamilySize` is calculated as `SibSp + Parch + 1`. `IsAlone` is set to 1 when the passenger travels alone and 0 otherwise.

## 4. Training and validation
The labelled data is divided into training and validation subsets using an 80:20 stratified split with `random_state=42`. Stratification preserves the target-class proportions across the two subsets.

## 5. Classification algorithms
Three classifiers are evaluated:
- Logistic Regression with `max_iter=1000`.
- Decision Tree with `max_depth=5` and `random_state=42`.
- Random Forest with 200 estimators, `max_depth=6`, and `random_state=42`.

## 6. Evaluation and model selection
The models are evaluated using accuracy, precision, recall, and F1-score. The recorded validation results are:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8045 | 0.7833 | 0.6812 | 0.7287 |
| Decision Tree | 0.7654 | 0.7547 | 0.5797 | 0.6557 |
| Random Forest | 0.8156 | 0.8750 | 0.6087 | 0.7179 |

Logistic Regression is selected for final prediction because it achieved the highest F1-score in the recorded validation results. Random Forest achieved the highest accuracy and precision, while Logistic Regression provided the strongest F1-score and recall.

## 7. Final prediction
After model selection, the selected pipeline is refitted using all 891 labelled training records. It is then used to predict the 418 records in `test.csv`. The resulting predictions are saved as `outputs/titanic_submission.csv`. These are local project predictions; no external leaderboard submission is claimed.

## 8. Reproducibility
The scripts expect `train.csv` and `test.csv` in a folder named `dataset`. Install the dependencies listed in `requirements.txt`, then run `python main.py`. Run `python model_comparison.py` to independently reproduce the validation comparison and its model-comparison chart.
