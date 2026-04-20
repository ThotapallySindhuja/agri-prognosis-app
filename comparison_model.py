import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Optional: install xgboost if not installed
# pip install xgboost
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("C:\\Android Projects\\crop_recommendation(APP)\\data\\plant_data.csv")

# Ensure lowercase columns
df.columns = df.columns.str.lower()

# Features and target
X = df[['n','p','k','ph','moisture','plant']]
y = df['health']

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf'))
    ]),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=42
    ),
    
    "XGBoost": XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
}

# Evaluate using cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=4)
    print(f"{name} Accuracy: {scores.mean()*100:.2f}%")