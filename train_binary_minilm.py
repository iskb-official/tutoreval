import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib   # <-- add this

# 1. Load data
df = pd.read_csv("MRBench_V2_binary.csv")
X_text = df["response"].fillna("")
y = df["ped_binary"]

# 2. Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 3. Sentence embeddings
model_name = "all-MiniLM-L6-v2"
encoder = SentenceTransformer(model_name)

print("Encoding responses...")
X_emb = encoder.encode(
    X_text.tolist(),
    batch_size=64,
    show_progress_bar=True
)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_emb, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# 5. XGBoost classifier
clf = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
)
clf.fit(X_train, y_train)

# 6. Evaluation
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=le.classes_))
auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc:.3f}")

# 7. Save model components
joblib.dump(clf, "xgb_ped_binary_minilm.joblib")
joblib.dump(le, "label_encoder_ped_binary.joblib")

# Just record the encoder name (MiniLM is loaded by name later)
with open("encoder_name.txt", "w", encoding="utf-8") as f:
    f.write(model_name)

print("Saved xgb_ped_binary_minilm.joblib, label_encoder_ped_binary.joblib and encoder_name.txt")
