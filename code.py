# project_cti_textclf.py
import warnings
warnings.filterwarnings("ignore")  # option: enlève les warnings sklearn (convergence/metrics)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

try:
    import joblib
except ImportError:
    joblib = None


# -----------------------------
# 1) Data loading
# -----------------------------
TRAIN_PATH = "cyber-threat-intelligence-splited_train.csv"
VAL_PATH   = "cyber-threat-intelligence-splited_validate.csv"
TEST_PATH  = "cyber-threat-intelligence-splited_test.csv"

def load_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df

train = load_split(TRAIN_PATH)
val   = load_split(VAL_PATH)
test  = load_split(TEST_PATH)

print("Shapes:", train.shape, val.shape, test.shape)
print("Nb classes (train):", train["label"].nunique())


# -----------------------------
# 2) Eval helpers
# -----------------------------
def eval_on(df: pd.DataFrame, model, name: str):
    pred = model.predict(df["text"])
    acc = accuracy_score(df["label"], pred)
    f1w = f1_score(df["label"], pred, average="weighted")
    f1m = f1_score(df["label"], pred, average="macro")
    print(f"\n=== {name} ===")
    print("Accuracy:", round(acc, 4))
    print("F1-weighted:", round(f1w, 4))
    print("F1-macro:", round(f1m, 4))
    return acc, f1w, f1m

def plot_confusion_matrix(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------
# 3) Baselines (standard solutions)
# -----------------------------
models = {
    "LR": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000)),
        ("clf", LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced", n_jobs=-1))
    ]),
    "LinearSVC": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000)),
        ("clf", LinearSVC(class_weight="balanced"))
    ]),
    "NB": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000)),
        ("clf", MultinomialNB())
    ])
}

baseline_results = []
for name, model in models.items():
    model.fit(train["text"], train["label"])
    acc, f1w, f1m = eval_on(val, model, name=f"{name} (VAL)")
    baseline_results.append([name, "baseline", acc, f1w, f1m])

baseline_df = pd.DataFrame(baseline_results, columns=["model", "stage", "val_acc", "val_f1w", "val_f1m"])
print("\nBaseline summary:\n", baseline_df)


# -----------------------------
# 4) GridSearch on fixed validation (PredefinedSplit)
# -----------------------------
X_trainval = pd.concat([train["text"], val["text"]], axis=0).reset_index(drop=True)
y_trainval = pd.concat([train["label"], val["label"]], axis=0).reset_index(drop=True)

test_fold = np.r_[-np.ones(len(train), dtype=int), np.zeros(len(val), dtype=int)]
ps = PredefinedSplit(test_fold)

# ---- GridSearch: Logistic Regression
pipe_lr = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced", n_jobs=-1))
])

param_grid_lr = {
    "tfidf__max_features": [20000, 30000],
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__min_df": [1, 3, 5],
    "tfidf__max_df": [0.7, 0.9, 1.0],
    "clf__C": [0.1, 1, 10],
}

grid_lr = GridSearchCV(
    pipe_lr,
    param_grid=param_grid_lr,
    scoring="f1_weighted",
    cv=ps,
    n_jobs=-1,
    verbose=1
)
grid_lr.fit(X_trainval, y_trainval)

print("\n=== GRID SEARCH LR ===")
print("Best params:", grid_lr.best_params_)
print("Best VAL f1_weighted:", round(grid_lr.best_score_, 4))
best_lr = grid_lr.best_estimator_

# ---- GridSearch: LinearSVC
pipe_svc = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC(class_weight="balanced"))
])

param_grid_svc = {
    "tfidf__max_features": [20000, 30000],
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__min_df": [1, 3, 5],
    "tfidf__max_df": [0.7, 0.9, 1.0],
    "clf__C": [0.1, 1, 10],
}

grid_svc = GridSearchCV(
    pipe_svc,
    param_grid=param_grid_svc,
    scoring="f1_weighted",
    cv=ps,
    n_jobs=-1,
    verbose=1
)
grid_svc.fit(X_trainval, y_trainval)

print("\n=== GRID SEARCH SVC ===")
print("Best params:", grid_svc.best_params_)
print("Best VAL f1_weighted:", round(grid_svc.best_score_, 4))
best_svc = grid_svc.best_estimator_


# -----------------------------
# 5) Final test evaluation (refit on train+val, then test)
# -----------------------------
candidates = [
    ("Best LR (GridSearch)", best_lr, grid_lr.best_score_),
    ("Best SVC (GridSearch)", best_svc, grid_svc.best_score_),
]

final_rows = []
for name, model, val_f1w in candidates:
    model.fit(X_trainval, y_trainval)
    pred_test = model.predict(test["text"])
    acc = accuracy_score(test["label"], pred_test)
    f1w = f1_score(test["label"], pred_test, average="weighted")
    f1m = f1_score(test["label"], pred_test, average="macro")
    final_rows.append([name, val_f1w, acc, f1w, f1m])

final_df = pd.DataFrame(final_rows, columns=["model", "val_f1w", "test_acc", "test_f1w", "test_f1m"])
print("\nFinal comparison:\n", final_df)

# pick best on test_f1w
best_name = final_df.sort_values("test_f1w", ascending=False).iloc[0]["model"]
best_model = best_svc if "SVC" in best_name else best_lr
print("\n>>> FINAL MODEL CHOSEN:", best_name)

# Detailed report for final model
best_model.fit(X_trainval, y_trainval)
final_pred = best_model.predict(test["text"])
print("\n=== FINAL TEST REPORT ===")
print("Accuracy:", round(accuracy_score(test["label"], final_pred), 4))
print("F1-weighted:", round(f1_score(test["label"], final_pred, average="weighted"), 4))
print("F1-macro:", round(f1_score(test["label"], final_pred, average="macro"), 4))
print("\nClassification report:\n")
print(classification_report(test["label"], final_pred, zero_division=0))

# Confusion matrix plot
labels_sorted = sorted(test["label"].unique())
plot_confusion_matrix(
    test["label"], final_pred, labels_sorted,
    title=f"Confusion Matrix - {best_name}",
    out_path="confusion_matrix_final.png"
)
print("\nSaved: confusion_matrix_final.png")

# Save outputs
final_df.to_csv("results_summary.csv", index=False)
print("Saved: results_summary.csv")

if joblib is not None:
    joblib.dump(best_model, "best_model.joblib")
    print("Saved: best_model.joblib")
else:
    print("joblib not installed -> model not saved")
