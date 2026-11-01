# Titanic Survival Prediction — Figure Interpretations

This README contains the interpretation of every figure included in the `figures/` directory. Each section corresponds to the PNG file with the same numbered name.

## Figure 1: Missing Values in Titanic Dataset

**File:** `figures/01_missing_values.png`

The training dataset contains 177 missing Age values (19.87%), 687 missing Cabin values (77.10%), and 2 missing Embarked values (0.22%). Cabin has a particularly high proportion of missing entries, so it is excluded from the selected model features. Age is handled using median imputation, while Embarked is handled using most-frequent imputation.

**Suggested caption:** Missing Values in Titanic Dataset.

## Figure 2: Overall Passenger Survival Distribution

**File:** `figures/02_survival_distribution.png`

The 891 training records contain 549 non-survivors (class 0) and 342 survivors (class 1). This corresponds to approximately 61.6% non-survivors and 38.4% survivors. The class distribution is not perfectly balanced, which supports using a stratified train-validation split to preserve similar class proportions in both subsets.

**Suggested caption:** Overall Passenger Survival Distribution.

## Figure 3: Survival Rate by Gender

**File:** `figures/03_survival_by_gender.png`

The observed female survival rate is approximately 0.74, compared with approximately 0.19 for male passengers. Gender therefore appears to be an informative predictor in this dataset. This is an association observed in the records and should not be interpreted as proof of causation.

**Suggested caption:** Survival Rate by Gender.

## Figure 4: Survival Rate by Passenger Class

**File:** `figures/04_survival_by_passenger_class.png`

First-class passengers have the highest observed survival rate (approximately 0.63), followed by second-class passengers (approximately 0.47). Third-class passengers have the lowest rate (approximately 0.24). Passenger class is therefore a useful predictive feature. The chart describes an association in the dataset, not a causal relationship.

**Suggested caption:** Survival Rate by Passenger Class.

## Figure 5: Age Distribution by Survival Status

**File:** `figures/05_age_distribution_by_survival.png`

The age distributions of survivors and non-survivors overlap considerably, with many observations concentrated among young adults. This suggests that Age alone cannot clearly distinguish the two survival classes. Since Age contains missing entries, median imputation is applied in the numerical preprocessing pipeline.

**Suggested caption:** Age Distribution by Survival Status.

## Figure 6: Correlation Heatmap of Titanic Features

**File:** `figures/06_correlation_heatmap.png`

The heatmap shows pairwise correlations among the displayed numerical variables. Pclass and Fare have a negative correlation of approximately -0.55. SibSp and Parch have a positive correlation of approximately 0.41. Survived has a positive correlation with Fare (approximately 0.26) and a negative correlation with Pclass (approximately -0.34). Correlation measures association and does not imply causation.

**Suggested caption:** Correlation Heatmap of Titanic Features.

## Figure 7: Comparison of Machine Learning Models

**File:** `figures/07_model_performance_comparison.png`

The chart compares Logistic Regression, Decision Tree, and Random Forest on the validation subset. Random Forest achieves the highest accuracy (0.8156) and precision (0.8750). Logistic Regression achieves the highest recall (0.6812) and F1-score (0.7287). Decision Tree records the lowest accuracy (0.7654) and F1-score (0.6557). Logistic Regression is selected for final prediction because it achieved the highest F1-score.

**Suggested caption:** Comparison of Machine Learning Models.

## Figure 8: Confusion Matrices for the Three Classifiers

**File:** `figures/08_confusion_matrices.png`

The matrices show actual classes against predicted classes for the 179-record validation subset. Logistic Regression: [[97, 13], [22, 47]]; Decision Tree: [[97, 13], [29, 40]]; Random Forest: [[104, 6], [27, 42]]. Random Forest produces the fewest false positives (6), while Logistic Regression identifies the largest number of actual survivors (47) and achieves the strongest recall and F1-score.

**Suggested caption:** Confusion Matrices for the Three Classifiers.

## Overall observations

The exploratory figures indicate that survival outcomes vary across gender and passenger class, while the dataset contains substantial missingness in the Cabin and Age columns. The model comparison shows that Logistic Regression provides the highest F1-score (0.7287), Random Forest provides the highest accuracy (0.8156) and precision (0.8750), and Decision Tree has the lowest accuracy and F1-score among the evaluated models.

## Notes

- The model metrics and confusion matrices correspond to the recorded stratified 80:20 validation split with `random_state=42`.
- The validation subset contains 179 records.
- Interpretations describe observed dataset patterns and model results; they do not establish causal relationships.
