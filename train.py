import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris dataset from CSV
df = pd.read_csv("iris.csv")

# Input features
X = df[[
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]]

# Target
y = df["species"]

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "iris_model.pkl")

print("Model trained successfully!")
print("Model saved as iris_model.pkl")