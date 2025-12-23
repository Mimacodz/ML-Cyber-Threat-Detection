import pandas as pd

print("hello world")
raw_path = "cyber-threat-intelligence_all.csv"
raw = pd.read_csv(raw_path)

print("RAW shape:", raw.shape)
print("RAW columns:", list(raw.columns))
raw.head(10)

print("\nMissing values per column:")
print(raw.isna().sum().sort_values(ascending=False))

print("\nNumber of duplicated rows:", raw.duplicated().sum())

# Duplicates by text only (useful in NLP)
if "text" in raw.columns:
    print("Duplicated 'text' rows:", raw.duplicated(subset=["text"]).sum())

raw[["text", "label"]].head(10)

clean = raw.copy()

# Keep only usable samples
clean = clean.dropna(subset=["text", "label"]).copy()
clean["text"] = clean["text"].astype(str).str.strip()
clean["label"] = clean["label"].astype(str).str.strip()

# Remove empty text
clean = clean[clean["text"].str.len() > 0].copy()

# Optional: remove exact duplicated text
clean = clean.drop_duplicates(subset=["text"]).copy()

print("CLEAN shape:", clean.shape)
clean.head(5)

counts = clean["label"].value_counts()
print("Number of classes:", counts.shape[0])
counts.head(10)

import pandas as pd

train = pd.read_csv("cyber-threat-intelligence-splited_train.csv").dropna(subset=["text","label"])
val   = pd.read_csv("cyber-threat-intelligence-splited_validate.csv").dropna(subset=["text","label"])
test  = pd.read_csv("cyber-threat-intelligence-splited_test.csv").dropna(subset=["text","label"])

# sécurité
for df in (train, val, test):
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

print("Shapes:", train.shape, val.shape, test.shape)
print("Nb classes:", train["label"].nunique())

import pandas as pd
import matplotlib.pyplot as plt

train_path = "cyber-threat-intelligence-splited_train.csv"
train = pd.read_csv(train_path).dropna(subset=["text", "label"]).copy()
train["text"] = train["text"].astype(str)
train["label"] = train["label"].astype(str)

print("Train shape:", train.shape)
train[["text", "label"]].head(5)

top10 = train["label"].value_counts().head(10)
top10

plt.figure(figsize=(10, 5))
top10.plot(kind="bar")
plt.title("Top 10 label distribution (Train)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

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

from sklearn.metrics import accuracy_score, f1_score, classification_report

def eval_on(df, model, name="model"):
    pred = model.predict(df["text"])
    acc = accuracy_score(df["label"], pred)
    f1w = f1_score(df["label"], pred, average="weighted")
    f1m = f1_score(df["label"], pred, average="macro")
    print(f"\n=== {name} ===")
    print("Accuracy:", round(acc, 4))
    print("F1-weighted:", round(f1w, 4))
    print("F1-macro:", round(f1m, 4))
    return acc, f1w, f1m

# Fit sur train, évalue sur val
results = {}
for name, model in models.items():
    model.fit(train["text"], train["label"])
    results[name] = eval_on(val, model, name=f"{name} (VAL)")

import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit

X_trainval = pd.concat([train["text"], val["text"]], axis=0).reset_index(drop=True)
y_trainval = pd.concat([train["label"], val["label"]], axis=0).reset_index(drop=True)

test_fold = np.r_[-np.ones(len(train), dtype=int), np.zeros(len(val), dtype=int)]
ps = PredefinedSplit(test_fold)

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
    n_jobs=-1
)

grid_lr.fit(X_trainval, y_trainval)

print("\n=== GRID SEARCH LR ===")
print("Best params:", grid_lr.best_params_)
print("Best VAL f1_weighted:", round(grid_lr.best_score_, 4))

best_model = grid_lr.best_estimator_

# Refit sur train+val puis test
best_model.fit(X_trainval, y_trainval)

print("\n=== FINAL TEST (Best LR) ===")
eval_on(test, best_model, name="Best LR (TEST)")

print("\nClassification report (TEST):")
test_pred = best_model.predict(test["text"])
print(classification_report(test["label"], test_pred, zero_division=0))

from sklearn.ensemble import VotingClassifier

# estimators doivent être "des modèles", pas des pipelines identiques.
# Ici on fait 3 pipelines différents.
lr = models["LR"]
svm = models["LinearSVC"]
nb  = models["NB"]

voting = VotingClassifier(
    estimators=[("lr", lr), ("svm", svm), ("nb", nb)],
    voting="hard"
)

voting.fit(train["text"], train["label"])
eval_on(val, voting, name="Voting (VAL)")

# si tu veux le test final, refit sur train+val
voting.fit(X_trainval, y_trainval)
eval_on(test, voting, name="Voting (TEST)")

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

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
    cv=ps,      # ton PredefinedSplit train/val
    n_jobs=-1
)

grid_svc.fit(X_trainval, y_trainval)
print("Best SVC params:", grid_svc.best_params_)
print("Best SVC VAL f1_weighted:", grid_svc.best_score_)

best_svc = grid_svc.best_estimator_   # si ton objet s'appelle grid_svc
# ou best_svc = grid.best_estimator_ si tu l'as appelé autrement

best_svc.fit(X_trainval, y_trainval)

from sklearn.metrics import accuracy_score, f1_score, classification_report

test_pred = best_svc.predict(test["text"])
print("=== FINAL TEST (Best SVC) ===")
print("Test Accuracy:", round(accuracy_score(test["label"], test_pred), 4))
print("Test F1-weighted:", round(f1_score(test["label"], test_pred, average="weighted"), 4))
print("Test F1-macro:", round(f1_score(test["label"], test_pred, average="macro"), 4))
print("\nClassification report:\n")
print(classification_report(test["label"], test_pred, zero_division=0))

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

tree_model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=30000)),
    ("clf", DecisionTreeClassifier(random_state=42))
])

tree_model.fit(train["text"], train["label"])
eval_on(val, tree_model, name="DecisionTree (VAL)")
