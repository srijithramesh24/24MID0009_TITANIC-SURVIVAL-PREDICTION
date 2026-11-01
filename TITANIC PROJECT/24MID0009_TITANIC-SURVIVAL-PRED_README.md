# Titanic Survival Prediction Using Machine Learning

**Predictive Analytics | Supervised Machine Learning | Binary Classification**

## Project Information

| Detail | Information |
|---|---|
| Project Title | Titanic Survival Prediction Using Machine Learning |
| Student Name | SRIJITH R |
| Registration Number | 24MID0009 |
| Institution | Vellore Institute of Technology (VIT), Vellore |
| Project Guide | Dr. DURGESH KUMAR |
| Designation | Assistant Professor Sr. Grade 1 |
| Domain | Machine Learning and Predictive Analytics |
| Programming Language | Python |

## 1. Project Overview

This project develops a supervised machine learning system to predict whether a passenger survived the Titanic disaster using passenger-related attributes. The task is binary classification: `Survived = 0` means the passenger did not survive, while `Survived = 1` means the passenger survived.

The workflow includes exploratory data analysis, missing-value treatment, feature engineering, preprocessing, classification, model evaluation, model selection, and test-set prediction generation.

## 2. Objectives

- Explore the Titanic passenger dataset and identify useful patterns.
- Inspect and handle missing values.
- Engineer family-related features.
- Prepare numerical and categorical variables for machine learning.
- Train and compare classification algorithms.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Select a model using validation F1-score.
- Generate predictions for unseen passenger records.

## 3. Dataset

| Dataset | Records | Description |
|---|---:|---|
| `train.csv` | 891 | Passenger records with survival labels |
| `test.csv` | 418 | Passenger records without survival labels |
| `gender_submission.csv` | 418 | Reference prediction file |

Selected features include `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`. Two additional features are engineered: `FamilySize` and `IsAlone`.

## 4. Technologies and Libraries

- **Python** – Main programming language
- **Jupyter Notebook** – Development environment
- **Pandas and NumPy** – Data handling and numerical operations
- **Matplotlib and Seaborn** – Data visualization
- **Scikit-learn** – Preprocessing, classification, and evaluation

## 5. Methodology

1. Load and inspect the datasets.
2. Perform exploratory data analysis.
3. Identify missing values.
4. Create `FamilySize` and `IsAlone` features.
5. Impute missing values, scale numerical features, and encode categorical features.
6. Split labelled data using a stratified 80:20 training-validation split.
7. Train three classification algorithms.
8. Compare validation performance.
9. Select the classifier with the highest F1-score.
10. Retrain the selected classifier on all labelled records and predict the test dataset.

### Preprocessing

- Numerical features: median imputation followed by standard scaling.
- Categorical features: most-frequent imputation followed by one-hot encoding.
- Preprocessing is integrated with classification using Scikit-learn pipelines.

## 6. Classification Algorithms

1. **Logistic Regression** – A linear classification algorithm that estimates the probability of survival.
2. **Decision Tree** – A tree-based classifier that learns decision rules from passenger attributes.
3. **Random Forest** – An ensemble classifier that combines multiple decision trees.

## 7. Validation Results

The models were evaluated on the 179-record validation set.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.45% | 78.33% | 68.12% | **72.87%** |
| Decision Tree | 76.54% | 75.47% | 57.97% | 65.57% |
| Random Forest | **81.56%** | **87.50%** | 60.87% | 71.79% |

### Model Selection

Logistic Regression was selected as the final model because it achieved the highest validation F1-score of **72.87%**. Random Forest achieved the highest accuracy and precision, while Logistic Regression achieved the highest recall and F1-score.

## 8. Final Prediction Output

The selected Logistic Regression pipeline was retrained using all 891 labelled training records and used to predict the 418 test records.

| Prediction | Count |
|---|---:|
| Non-survivors (`0`) | 259 |
| Survivors (`1`) | 159 |
| **Total** | **418** |

The generated prediction file is `titanic_submission.csv`. These are local test-set predictions; no external leaderboard score is claimed.

## 9. Repository Structure

```text
TITANIC PROJECT/
├── 24MID0009_TITANIC-SURVIVAL-PRED_README.md
├── requirements.txt
├── .gitignore
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── notebooks/
│   └── Titanic_Survival_Prediction.ipynb
├── outputs/
│   └── titanic_submission.csv
├── figures/
│   └── (project visualizations & Inetpretations readme.md)
└── report/
    └── Titanic_Survival_Prediction_Report.pdf
```

This is a recommended structure. Include only files that are actually present in your repository.

## 10. Installation and Execution

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook in the `notebooks/` directory. Ensure dataset paths in the notebook match the repository structure, then execute the cells in order.

## 11. Limitations and Future Scope

The reported results are based on one stratified train-validation split and predefined model parameters. Further work could include cross-validation, hyperparameter optimization, additional feature engineering, and evaluation on independent labelled data. The historical dataset also contains missing information and should not be treated as representative of modern maritime incidents.

## 12. Student and Faculty Details

**Developed by**  
**SRIJITH R**  
Registration Number: 24MID0009  
Vellore Institute of Technology (VIT), Vellore

**Under the Guidance of**  
**Dr. DURGESH KUMAR**  
Assistant Professor Sr. Grade 1  
Vellore Institute of Technology (VIT), Vellore

---

*Academic Machine Learning Project — VIT Vellore*
