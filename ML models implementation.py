# ============================================================
# 0. Imports
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt


# ============================================================
# 1. Load data
#    -> assumes a CSV with at least: "text" and "label" or "label_encoded"
# ============================================================

DATA_PATH = "cyber-threat-intelligence_clean.csv"  # change if needed
df = pd.read_csv(DATA_PATH)

# If label is not encoded yet, encode it
if "label_encoded" not in df.columns:
    print("label_encoded not found, encoding label...")
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
else:
    print("Using existing label_encoded column.")

X_text = df["text"]
y = df["label_encoded"]

# Train / test split (stratified because of class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))


# Small helper to print metrics
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    return y_pred


# ============================================================
# 2. Baseline model: TF-IDF + Logistic Regression
# ============================================================

baseline_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 1)   # unigrams only for baseline
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    ))
])

print("\nTraining baseline Logistic Regression...")
baseline_pipe.fit(X_train, y_train)
y_pred_baseline = evaluate_model("Baseline Logistic Regression", baseline_pipe, X_test, y_test)


# ============================================================
# 3. Tuned Logistic Regression (GridSearchCV)
#    -> hyperparameter tuning
# ============================================================

tuned_pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))
])

param_grid_lr = {
    "tfidf__max_df": [0.7, 0.9, 1.0],
    "tfidf__min_df": [1, 5],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1.0, 10.0]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_lr = GridSearchCV(
    estimator=tuned_pipe,
    param_grid=param_grid_lr,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1
)

print("\nRunning GridSearchCV for Logistic Regression...")
grid_lr.fit(X_train, y_train)

print("\nBest parameters (LogReg):", grid_lr.best_params_)
print("Best CV weighted F1:", grid_lr.best_score_)

best_lr = grid_lr.best_estimator_
y_pred_best_lr = evaluate_model("Tuned Logistic Regression", best_lr, X_test, y_test)


# ============================================================
# 4. Decision Tree model (to discuss overfitting/underfitting)
# ============================================================

dt_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", DecisionTreeClassifier(random_state=42))
])

param_grid_dt = {
    "clf__max_depth": [5, 10, 20, None],
    "clf__min_samples_split": [2, 10, 20]
}

grid_dt = GridSearchCV(
    estimator=dt_pipe,
    param_grid=param_grid_dt,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1
)

print("\nRunning GridSearchCV for Decision Tree...")
grid_dt.fit(X_train, y_train)

print("\nBest parameters (Decision Tree):", grid_dt.best_params_)
print("Best CV weighted F1 (tree):", grid_dt.best_score_)

best_dt = grid_dt.best_estimator_
y_pred_best_dt = evaluate_model("Tuned Decision Tree", best_dt, X_test, y_test)


# ============================================================
# 5. Dimension reduction: TF-IDF -> TruncatedSVD -> Logistic Regression
#    (PCA-style for text)
# ============================================================

svd_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000)),
    ("svd", TruncatedSVD(n_components=100, random_state=42)),
    ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))
])

print("\nTraining Logistic Regression with TruncatedSVD (dimension reduction)...")
svd_pipe.fit(X_train, y_train)
y_pred_svd = evaluate_model("Logistic Regression + SVD (100 components)", svd_pipe, X_test, y_test)


# ============================================================
# 6. Confusion matrix for the best model (tuned Logistic Regression)
# ============================================================

print("\nPlotting confusion matrix for tuned Logistic Regression...")
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best_lr, ax=ax)
plt.title("Confusion matrix – Tuned Logistic Regression")
plt.tight_layout()
plt.show()
