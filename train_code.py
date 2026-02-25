# ─────────────────────────────────────────────
#  PART 1 – MODEL TRAINING
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import randint, uniform
import joblib

# ── Load & clean ──────────────────────────────
churn = pd.read_csv(r'D:\project 2026\telecom Customer churn\Customer-Churn (1).csv')
new_churn = churn.copy()

new_churn['TotalCharges'] = pd.to_numeric(new_churn['TotalCharges'], errors='coerce')
new_churn.dropna(how='any', inplace=True)

# ── Tenure grouping ───────────────────────────
labels = ["{0}-{1}".format(i, i + 11) for i in range(1, 72, 12)]
new_churn['tenure_group'] = pd.cut(
    new_churn.tenure, range(1, 80, 12), right=False, labels=labels
)
new_churn.drop(columns=['customerID', 'tenure'], inplace=True)

# ── Features / target ─────────────────────────
X = new_churn.drop(columns='Churn')
y = new_churn['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ── Pipeline ──────────────────────────────────
num_f = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_f = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_f),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_f)
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('Classifier', AdaBoostClassifier())
])

full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
print("Baseline accuracy:", full_pipeline.score(X_test, y_test))
print(classification_report(y_test, y_pred))

# ── Hyperparameter tuning ─────────────────────
param_distributions = {
    'Classifier__n_estimators': randint(50, 200),
    'Classifier__learning_rate': uniform(0.01, 2.0),
}

random_search = RandomizedSearchCV(
    estimator=full_pipeline,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='f1',
    verbose=2,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best F1 Score  :", random_search.best_score_)

best_model = random_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print('\nTest classification report:\n', classification_report(y_test, y_pred_best))

joblib.dump(best_model, 'best_ada_model.pkl')
print('Best model saved successfully')