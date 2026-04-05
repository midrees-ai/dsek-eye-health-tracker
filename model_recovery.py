import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load preprocessing data
from preprocessing import X_train2, X_test2, y_rec_train, y_rec_test

# Build model 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train2, y_rec_train)

# Predict
y_pred = model.predict(X_test2)

# Results
print(f"Recovery Stage Model Accuracy: {accuracy_score(y_rec_test, y_pred):.2f}")
print("\nClassification_report:")
print(classification_report(y_rec_test, y_pred))

# Save model 
joblib.dump(model, 'recovery_model.pkl')
print("\nModel saved as recovery_model.pkl")
