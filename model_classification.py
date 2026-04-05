import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load preprocessed data
from preprocessing import X_train, X_test, y_iop_train, y_iop_test

# Build model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_iop_train)

# Predict
y_pred = model.predict(X_test)

# Results
print(f"\nIOP Risk Model Accuracy: {accuracy_score(y_iop_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_iop_test, y_pred))

# Save model
joblib.dump(model, 'iop_risk_model.pkl')
print("\nModel saved as iop_risk_model.pkl")

