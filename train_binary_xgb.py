import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# 1. Load binary data (Good vs Poor)
df = pd.read_csv("MRBench_V2_binary.csv")

X_text = df["response"].fillna("")
y = df["ped_binary"]

# 2. Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)   # e.g., Good/Poor -> 0/1 (order is alphabetical)

# 3. TF-IDF features
tfidf = TfidfVectorizer(
    max_features=40000,
    ngram_range=(1, 2),
    min_df=2
)
X = tfidf.fit_transform(X_text)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# 5. XGBoost classifier
clf = XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
)

clf.fit(X_train, y_train)

# 6. Evaluation
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=le.classes_))
print("AUC:", roc_auc_score(y_test, y_prob))

# 7. Export top 50 features by importance (for explanations)
feature_names = tfidf.get_feature_names_out()
importances = clf.feature_importances_

top_idx = np.argsort(importances)[-50:][::-1]
top_feats = [(feature_names[i], float(importances[i])) for i in top_idx]

df_feats = pd.DataFrame(top_feats, columns=["feature", "importance"])
df_feats.to_csv("top50_tfidf_xgb_features.csv", index=False)

print("\nSaved top50_tfidf_xgb_features.csv")
print(df_feats.head(10))
