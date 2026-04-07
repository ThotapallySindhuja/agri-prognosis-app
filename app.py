import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score ;

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/plant_data.csv")

# Features (plant is now an input)
X = df[['n','p','k','ph','moisture','plant']]

# Target
y = df['health']

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# Train Model
# ==========================

health_model = RandomForestClassifier(n_estimators=100, random_state=42)

health_model.fit(X_train_scaled, y_train)

# ==========================
# Evaluate Model
# ==========================

train_acc = accuracy_score(y_train, health_model.predict(X_train_scaled))
test_acc = accuracy_score(y_test, health_model.predict(X_test_scaled))

print(f"Health Model - Train Accuracy: {train_acc:.4f}")
print(f"Health Model - Test Accuracy: {test_acc:.4f}")

# ==========================
# Save Model
# ==========================

pickle.dump(health_model, open("health_model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))

# Load model
health_model = pickle.load(open("health_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# ==========================
# Flask App
# ==========================

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():

    n = float(request.form['n'])
    p = float(request.form['p'])
    k = float(request.form['k'])
    ph = float(request.form['ph'])
    moisture = float(request.form['moisture'])
    plant = int(request.form['plant']) 

    features = np.array([[n,p,k,ph,moisture,plant]]) 
    features_scaled = scaler.transform(features)

    health_pred = health_model.predict(features_scaled)[0]

    health_status = "Healthy 🌱" if health_pred == 1 else "Unhealthy ⚠️"

    if health_pred == 0:
        suggestion = "Check soil nutrients and add fertilizer if required."
    else:
        suggestion = "Soil conditions are good."

    return render_template(
        "index.html",
        health_result=f"Plant Health: {health_status}",
        accuracy=f"Model Accuracy: {test_acc*100:.2f}%",
        suggestion=suggestion,
        n=n,
        p=p,
        k=k,
        ph=ph,  
        moisture=moisture,
        plant=plant
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)   