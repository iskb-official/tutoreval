# train_cv_robust.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             confusion_matrix, roc_curve, precision_recall_curve, auc)

# 1. Load Data
print("Loading data...")
df = pd.read_csv("MRBench_V2_binary.csv")
df["response"] = df["response"].fillna("")
X_text = df["response"]
y = df["ped_binary"].map({"Good": 1, "Poor": 0}).values

# IMPORTANT: To prevent data leakage, we group by the original MRBench seed ID.
# To this:
if "conversation_id" not in df.columns:
    raise KeyError("A 'conversation_id' column is required to run StratifiedGroupKFold...")
groups = df["conversation_id"].values

# 2. Setup Grouped Cross-Validation (5 Folds)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = []
f1_scores = []
auc_scores = []

# Lists to store ALL predictions for global plotting
all_y_test = []
all_y_pred = []
all_y_proba = []

print(f"Starting 5-Fold Grouped CV on {len(df)} samples...")

fold = 1
# Note the addition of groups=groups in the split function
for train_index, test_index in sgkf.split(X_text, y, groups=groups):
    X_train_txt, X_test_txt = X_text.iloc[train_index], X_text.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 3. Vectorization (Fit ONLY on training fold to avoid data leakage)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train_txt)
    X_test_vec = vectorizer.transform(X_test_txt)

    # 4. Train Gradient Boosting Classifier
    # Calculate weights to handle class imbalance
    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    weight_pos = n_neg / n_pos if n_pos > 0 else 1.0
    
    sample_weights = np.ones_like(y_train, dtype=float)
    sample_weights[y_train == 1] = weight_pos

    # Note: You can swap this for XGBClassifier if you have xgboost installed
    clf = GradientBoostingClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1, 
        random_state=42
    )
    clf.fit(X_train_vec, y_train, sample_weight=sample_weights)

    # 5. Evaluate
    y_pred = clf.predict(X_test_vec)
    y_proba = clf.predict_proba(X_test_vec)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        auc_val = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc_val = 0.5 # Fallback for edge cases

    acc_scores.append(acc)
    f1_scores.append(f1)
    auc_scores.append(auc_val)

    # Store for global plotting
    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)

    print(f"Fold {fold}: Acc={acc:.3f} | F1={f1:.3f} | AUC={auc_val:.3f}")
    fold += 1

# 6. Final Robust Metrics for Paper
print("\n=== ROBUST PERFORMANCE (For Paper) ===")
print(f"Mean Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
print(f"Mean F1 Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
print(f"Mean AUC:      {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")

# --- PLOTTING ---

# 1. Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(all_y_test, all_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Poor', 'Good'], yticklabels=['Poor', 'Good'])
plt.title('Cross-Validated Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('robust_confusion_matrix.png')
print("\nSaved robust_confusion_matrix.png")

# 2. ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(all_y_test, all_y_proba)
roc_auc_val = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Cross-Validated ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('robust_roc_curve.png')
print("Saved robust_roc_curve.png")

# 3. Precision-Recall Curve
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(all_y_test, all_y_proba)
pr_auc_val = auc(recall, precision)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc_val:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Cross-Validated Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig('robust_pr_curve.png')
print("Saved robust_pr_curve.png")

# 7. Train Final Production Model
print("\nTraining final production model on full dataset...")
final_vec = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
X_full = final_vec.fit_transform(X_text)

# Final weights for full training
n_pos_full = sum(y)
n_neg_full = len(y) - n_pos_full
weight_pos_full = n_neg_full / n_pos_full if n_pos_full > 0 else 1.0
sample_weights_full = np.ones_like(y, dtype=float)
sample_weights_full[y == 1] = weight_pos_full

final_clf = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
)
final_clf.fit(X_full, y, sample_weight=sample_weights_full)

joblib.dump(final_clf, "xgb_ped_binary_v4_robust.joblib")
joblib.dump(final_vec, "tfidf_ped_binary_v4_robust.joblib")
print("Saved v4_robust models.")