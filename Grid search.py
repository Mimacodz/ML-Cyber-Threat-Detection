# ============================================================
# 0. Imports
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# ============================================================
# 1. Load project data and prepare labels
# ============================================================

DATA_PATH = "cyber-threat-intelligence_clean.csv"  # adapt the path if needed
df = pd.read_csv(DATA_PATH)

# Make sure we have numeric labels
if "label_encoded" not in df.columns:
    print("label_encoded not found, encoding 'label'...")
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
else:
    print("Using existing label_encoded column.")
    le = None  # optional

X_text = df["text"]
y = df["label_encoded"]

# Train/test split (same idea as in the lab)
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))


# ============================================================
# 2. Vectorize text with TF-IDF (one time for all models)
# ============================================================

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)  # unigrams + bigrams
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

print("TF-IDF shape (train):", X_train_vec.shape)


# Small helper to evaluate models
def evaluate_model(name, model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    return acc, y_pred


# ============================================================
# 3. Baseline models + Grid Search (Part II: apply grid search)
# ============================================================

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 3.1 Logistic Regression
print("\n>>> Grid Search: Logistic Regression")

lr = LogisticRegression(max_iter=1000, n_jobs=-1)
param_grid_lr = {
    "C": [0.1, 1.0, 10.0]
}

grid_lr = GridSearchCV(
    estimator=lr,
    param_grid=param_grid_lr,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1
)

grid_lr.fit(X_train_vec, y_train)
best_lr = grid_lr.best_estimator_
print("Best params (LR):", grid_lr.best_params_)
print("Best CV F1 (LR):", grid_lr.best_score_)

acc_lr, y_pred_lr = evaluate_model("Best Logistic Regression", best_lr, X_test_vec, y_test)


# 3.2 SVM (LinearSVC)
print("\n>>> Grid Search: Linear SVM")

svm = LinearSVC()
param_grid_svm = {
    "C": [0.1, 1.0, 10.0]
}

grid_svm = GridSearchCV(
    estimator=svm,
    param_grid=param_grid_svm,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1
)

grid_svm.fit(X_train_vec, y_train)
best_svm = grid_svm.best_estimator_
print("Best params (SVM):", grid_svm.best_params_)
print("Best CV F1 (SVM):", grid_svm.best_score_)

acc_svm, y_pred_svm = evaluate_model("Best Linear SVM", best_svm, X_test_vec, y_test)


# 3.3 Decision Tree
print("\n>>> Grid Search: Decision Tree")

dt = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_dt = GridSearchCV(
    estimator=dt,
    param_grid=param_grid_dt,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1
)

grid_dt.fit(X_train_vec, y_train)
best_dt = grid_dt.best_estimator_
print("Best params (DT):", grid_dt.best_params_)
print("Best CV F1 (DT):", grid_dt.best_score_)

acc_dt, y_pred_dt = evaluate_model("Best Decision Tree", best_dt, X_test_vec, y_test)


# ============================================================
# 4. Ensemble models: Bagging (Part II: apply ensemble methods)
# ============================================================

# 4.1 Bagging SVM
print("\n>>> Bagging: SVM")

bag_svm = BaggingClassifier(
    base_estimator=best_svm,
    n_estimators=10,
    max_samples=0.8,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

bag_svm.fit(X_train_vec, y_train)
acc_bag_svm, y_pred_bag_svm = evaluate_model("Bagging SVM", bag_svm, X_test_vec, y_test)


# 4.2 Bagging Decision Tree
print("\n>>> Bagging: Decision Tree")

bag_dt = BaggingClassifier(
    base_estimator=best_dt,
    n_estimators=10,
    max_samples=0.8,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

bag_dt.fit(X_train_vec, y_train)
acc_bag_dt, y_pred_bag_dt = evaluate_model("Bagging Decision Tree", bag_dt, X_test_vec, y_test)


# ============================================================
# 5. Voting classifier (hybrid ensemble)
#    We use soft voting with LR + Decision Tree (both have predict_proba)
# ============================================================

print("\n>>> Voting Classifier: LR + Decision Tree (soft voting)")

# Make sure both models support predict_proba
# LogisticRegression and DecisionTreeClassifier do.
voting_clf = VotingClassifier(
    estimators=[
        ("lr", best_lr),
        ("tree", best_dt)
    ],
    voting="soft"
)

voting_clf.fit(X_train_vec, y_train)
acc_vote, y_pred_vote = evaluate_model("Voting (LR + DT, soft)", voting_clf, X_test_vec, y_test)


# ============================================================
# 6. Confusion matrices for a few key models
# ============================================================

print("\nPlotting confusion matrices...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_lr, ax=axes[0]
)
axes[0].set_title("Confusion matrix – Logistic Regression")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_bag_svm, ax=axes[1]
)
axes[1].set_title("Confusion matrix – Bagging SVM")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_vote, ax=axes[2]
)
axes[2].set_title("Confusion matrix – Voting (LR + DT)")

plt.tight_layout()
plt.show()


# ============================================================
# 7. Accuracy comparison bar chart
# ============================================================

model_names = [
    "LogReg",
    "SVM",
    "DecisionTree",
    "Bagging SVM",
    "Bagging DT",
    "Voting"
]

accuracies = [
    acc_lr,
    acc_svm,
    acc_dt,
    acc_bag_svm,
    acc_bag_dt,
    acc_vote
]

plt.figure(figsize=(8, 4))
plt.bar(model_names, accuracies)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.title("Model accuracy comparison (project dataset)")
plt.tight_layout()
plt.show()
