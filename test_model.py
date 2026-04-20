import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==========================================
# Define Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "plant_data.csv")

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
    C=0.15,
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

print(f"SVM Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"SVM Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
