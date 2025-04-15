import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings

import os
os.environ['XGB_WARN_NO_USE'] = '0'  # Suppress XGBoost warnings


# Load Dataset
df = pd.read_csv("gym_recommendation.csv")  

# Define Input (X) and Output (y)
X = df[['Height', 'Weight', 'Sex', 'Hypertension', 'Diabetes', 'Fitness Goal']]  # Features
y = df[['Level', 'Fitness Type', 'Exercises', 'Diet', 'Recommendation']] # Categorical Outputs

# Convert Input Features (X) - One-Hot Encoding for Categorical Features
encoder = OneHotEncoder(drop='first', sparse_output=False)  # Updated for sklearn v1.2+
X_encoded = encoder.fit_transform(X[['Sex', 'Hypertension', 'Diabetes', 'Fitness Goal']])
X_numeric = np.hstack((X[['Height', 'Weight']].values, X_encoded))  # Combine with numeric features

# Convert Output Labels (y) - Label Encoding
label_encoders = {}  # Store label encoders for decoding later
y_encoded = pd.DataFrame()

for col in y.columns:
    le = LabelEncoder()
    y_encoded[col] = le.fit_transform(y[col])  # Convert text to numbers
    label_encoders[col] = le  # Store encoder for later decoding

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_encoded, test_size=0.2, random_state=42)

# Define Models
models = {
    "Random Forest": MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
    "Logistic Regression": MultiOutputClassifier(LogisticRegression(max_iter=500)),
    "SVM": MultiOutputClassifier(SVC()),
    "KNN": MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5)),
    "Gradient Boosting": MultiOutputClassifier(GradientBoostingClassifier(n_estimators=100, random_state=42)),
    "XGBoost": MultiOutputClassifier(XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
}

# Train and Evaluate Models
best_model = None
best_accuracy = 0
model_accuracies = {}

print("Evaluating Models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.mean([accuracy_score(y_test[col], y_pred[:, i]) for i, col in enumerate(y_test.columns)])
    model_accuracies[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Save Best Model & Encoders
with open("models.pkl", "wb") as model_file:
    pickle.dump((best_model, encoder, label_encoders), model_file)

print("Model trained and saved successfully!")
