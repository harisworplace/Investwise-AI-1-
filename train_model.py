import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("startup_valuation_dataset.csv")

# -----------------------------
# Keep Required Columns
# -----------------------------
df = df[
    [
        "industry",
        "country",
        "region",
        "funding_round",
        "funding_amount_usd",
        "employee_count",
        "estimated_revenue_usd",
        "estimated_valuation_usd",
        "exited"
    ]
]

# Remove missing values
df = df.dropna()

# -----------------------------
# Encode Categorical Columns
# -----------------------------
encoders = {}

for col in ["industry", "country", "region", "funding_round"]:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

# Encode Target
target_encoder = LabelEncoder()
df["exited"] = target_encoder.fit_transform(df["exited"])

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("exited", axis=1)
y = df["exited"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------
# Decision Tree
# -----------------------------
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy :", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, prediction))

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/investment_model.pkl")
joblib.dump(encoders, "models/label_encoders.pkl")
joblib.dump(target_encoder, "models/target_encoder.pkl")

print("\nModel Saved Successfully!")