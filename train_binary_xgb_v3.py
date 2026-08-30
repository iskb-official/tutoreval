# train_binary_xgb_v3.py (CALIBRATED)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib

df = pd.read_csv("MRBench_V2_binary_v2.csv")
X_text = df["response"].fillna("")
y = df["ped_binary_v2"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

tfidf = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2)
X = tfidf.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# BETTER XGB + CALIBRATION
base_clf = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=1.4,  # Balance Good class
    random_state=42
)

calibrated_clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=5)
calibrated_clf.fit(X_train, y_train)

y_pred = calibrated_clf.predict(X_test)
y_prob = calibrated_clf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=le.classes_))
print("AUC:", roc_auc_score(y_test, y_prob))

# Save v3
joblib.dump(calibrated_clf, "xgb_ped_binary_v3_calibrated.joblib")
joblib.dump(tfidf, "tfidf_ped_binary_v3.joblib")
joblib.dump(le, "label_encoder_ped_binary_v3.joblib")
