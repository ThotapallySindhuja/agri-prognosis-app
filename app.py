import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==========================================
# Initialize Flask App
# ==========================================
app = Flask(__name__)

# ==========================================
# Define Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "plant_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "health_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = df.columns.str.lower()

# Remove missing & duplicate values
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# ==========================================
# Features and Target
# ==========================================
X = df[['n', 'p', 'k', 'ph', 'moisture', 'plant']]
y = df['health']

# ==========================================
# Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================================
# Feature Scaling (IMPORTANT for SVM)
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Train SVM Model
# ==========================================
health_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma=0.01
)

health_model.fit(X_train_scaled, y_train)

# ==========================================
# Evaluate Model
# ==========================================
train_acc = accuracy_score(
    y_train, health_model.predict(X_train_scaled)
)
test_acc = accuracy_score(
    y_test, health_model.predict(X_test_scaled)
)

print(f"SVM Train Accuracy: {train_acc:.4f}")
print(f"SVM Test Accuracy: {test_acc:.4f}")

# ==========================================
# Save Model & Scaler
# ==========================================
with open(MODEL_PATH, "wb") as f:
    pickle.dump(health_model, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

# Load again (for safety)
with open(MODEL_PATH, "rb") as f:
    health_model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ==========================================
# Flask Routes
# ==========================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        accuracy=f"Model Accuracy: {test_acc * 100:.2f}%"
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs
        n = float(request.form["n"])
        p = float(request.form["p"])
        k = float(request.form["k"])
        ph = float(request.form["ph"])
        moisture = float(request.form["moisture"])
        plant = int(request.form["plant"])

        # Create dataframe
        features = pd.DataFrame(
            [[n, p, k, ph, moisture, plant]],
            columns=['n', 'p', 'k', 'ph', 'moisture', 'plant']
        )

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        health_pred = health_model.predict(features_scaled)[0]

        # Output
        if health_pred == 1:
            health_status = "Healthy 🌱"
            suggestion = "Soil conditions are good."
        else:
            health_status = "Unhealthy ⚠️"
            suggestion = "Check soil nutrients and add fertilizer if required."

        return render_template(
            "index.html",
            health_result=f"Plant Health: {health_status}",
            accuracy=f"Model Accuracy: {test_acc * 100:.2f}%",
            suggestion=suggestion,
            n=n,
            p=p,
            k=k,
            ph=ph,
            moisture=moisture,
            plant=plant
        )

    except Exception as e:
        return render_template(
            "index.html",
            health_result="Error occurred during prediction.",
            suggestion=str(e),
            accuracy=f"Model Accuracy: {test_acc * 100:.2f}%"
        )
'''print("INPUT:", features)
print("PRED:", health_pred)
'''
# ==========================================
# Run App
# ==========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)