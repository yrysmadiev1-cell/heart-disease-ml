# Heart Disease Prediction with Machine Learning

End-to-end binary classification project for predicting the presence of heart disease from patient clinical attributes.

## Project goal

The goal is to predict the binary target:

- `0` — no heart disease
- `1` — heart disease

The project covers exploratory data analysis, model comparison, hyperparameter tuning, cross-validation, model evaluation, feature importance, and error analysis.

## Dataset

The dataset contains **303 observations**, **13 input features**, and one target variable.

Main features include:

- `age` — age
- `sex` — sex
- `cp` — chest pain type
- `trestbps` — resting blood pressure
- `chol` — serum cholesterol
- `fbs` — fasting blood sugar
- `restecg` — resting ECG result
- `thalach` — maximum heart rate achieved
- `exang` — exercise-induced angina
- `oldpeak` — ST depression induced by exercise
- `slope` — slope of the peak exercise ST segment
- `ca` — number of major vessels
- `thal` — thallium stress test result

## Workflow

1. Problem definition and target analysis
2. Exploratory data analysis
3. Missing-value and class-balance checks
4. Correlation analysis
5. Train/test split with stratification
6. Baseline model comparison
7. Scaling with `Pipeline` for Logistic Regression and KNN
8. Hyperparameter tuning with `RandomizedSearchCV` and `GridSearchCV`
9. Final evaluation of the best Random Forest model
10. Cross-validation
11. Feature importance
12. Error analysis
13. Model persistence with `joblib`

## Models

The following models were compared:

- Logistic Regression
- K-Nearest Neighbors
- Random Forest

Final held-out test accuracy:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 0.7869 |
| KNN | 0.7869 |
| Random Forest | **0.8197** |

The tuned Random Forest was selected for detailed evaluation.

## Final Random Forest metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.8197 |
| Precision | 0.7619 |
| Recall | **0.9697** |
| F1-score | 0.8533 |
| ROC-AUC | **0.9069** |

Confusion matrix:

```text
[[18, 10],
 [ 1, 32]]
```

The model detected 32 of 33 positive cases in the held-out test set, producing one false negative and ten false positives.


## Visual results

### Model comparison

![Model comparison](reports/figures/model_comparison.png)

### Confusion matrix

![Confusion matrix](reports/figures/confusion_matrix.png)

### ROC curve

![ROC curve](reports/figures/roc_curve.png)

### Feature importance

![Feature importance](reports/figures/feature_importance.png)

## Cross-validation

5-fold stratified cross-validation on the training data:

| Metric | Mean | Std |
|---|---:|---:|
| Accuracy | 0.8180 | 0.0971 |
| Precision | 0.8172 | 0.0950 |
| Recall | 0.8632 | 0.0868 |
| F1-score | 0.8386 | 0.0867 |

The variation between folds indicates that the dataset is small and model performance depends noticeably on the particular split.

## Feature importance

Top Random Forest features:

| Feature | Importance |
|---|---:|
| `cp` | 0.1978 |
| `thal` | 0.1634 |
| `oldpeak` | 0.1199 |
| `ca` | 0.1028 |
| `thalach` | 0.1015 |
| `exang` | 0.0951 |

Feature importance describes how useful a feature was to the fitted Random Forest; it should not be interpreted as causality.

## Repository structure

```text
heart-disease-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── heart-disease.csv
├── notebooks/
│   └── heart_disease_analysis.ipynb
├── src/
│   └── train.py
└── models/
    └── heart_disease_random_forest.joblib
```

## Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd heart-disease-ml

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Open the notebook:

```bash
jupyter notebook notebooks/heart_disease_analysis.ipynb
```

Or train/evaluate the final model from the command line:

```bash
python src/train.py
```

## Notes

This is a portfolio/educational machine-learning project. It is not intended for clinical use or medical decision-making.

## Possible improvements

- Evaluate on a larger independent dataset
- Compare with XGBoost or CatBoost
- Use nested cross-validation for stricter model-selection evaluation
- Add permutation importance or SHAP
- Calibrate predicted probabilities
- Perform threshold selection using validation data rather than the final test set
