# =========================
# train_model.py
# =========================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
df = pd.read_excel("loan_approval_dataset_5000.xlsx")
# df = pd.read_excel(r"C:\Users\admin\Downloads\loan_approval_dataset_5000.xlsx")


# Encode Employment Type
encoder = LabelEncoder()
df["EmploymentType"] = encoder.fit_transform(df["EmploymentType"])

# Features and Target
X = df.drop("LoanApproved", axis=1)
y = df["LoanApproved"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Save Model
joblib.dump(model, "loan_model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("Model Saved Successfully")