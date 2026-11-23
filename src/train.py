import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.data_prep import load_and_inspect, clean_and_engineer_features
def train_and_evaluate(df: pd.DataFrame) -> None:
    X = df.drop(columns = ["Machine failure"])
    y = df["Machine failure"]

    X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify = y)

    model = RandomForestClassifier(n_estimators = 100, class_weight =  "balanced", random_state = 42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_probs = model.predict_proba(X_test)[:, 1]
    threshold=0.50
    y_pred_tuned = (y_probs >= threshold).astype(int)
    
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred_tuned)}")
    print(f"Classification report: \n {classification_report(y_test, y_pred_tuned)}")
    print(f"\nFeature Importances:\n{importances}")


    artifact ={
        "model": model,
        "threshold": threshold,
        "features": X.columns.tolist()
    }
    joblib.dump(artifact, "models/rf_predictive_maintenance.joblib")

if __name__ == "__main__":
    df = load_and_inspect("data/ai4i2020.csv")
    df = clean_and_engineer_features(df)
    train_and_evaluate(df)
    
