import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

DATA_FILE = 'landmarks.csv'
MODEL_FILE = 'model.pkl'

if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found. Please run collect_data.py first.")
    exit()

print("Loading data...")
df = pd.read_csv(DATA_FILE)

if len(df) < 10:
    print("Warning: Very little data available. Model might not train well.")

# Separate features (X) and labels (y)
X = df.drop('label', axis=1)
y = df['label']

print(f"Dataset shape: {X.shape}. Found classes: {y.unique()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
print(f"\nSaving model to {MODEL_FILE}...")
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)

print("Done!")
