import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model and dataset
model = joblib.load("iris_model.pkl")
df = pd.read_csv("iris.csv")

st.title("🌸 Iris Flower Prediction")

# Input values
sl = st.number_input("Sepal Length (cm)", min_value=0.0)
sw = st.number_input("Sepal Width (cm)", min_value=0.0)
pl = st.number_input("Petal Length (cm)", min_value=0.0)
pw = st.number_input("Petal Width (cm)", min_value=0.0)

if st.button("Predict"):
    prediction = model.predict([[sl, sw, pl, pw]])
    st.success(f"Predicted Flower: {prediction[0].title()}")

# Graph
st.subheader("📊 Iris Dataset Graph")

fig, ax = plt.subplots()

for species in df["species"].unique():
    data = df[df["species"] == species]
    ax.scatter(
        data["sepal_length"],
        data["petal_length"],
        label=species
    )

ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Petal Length (cm)")
ax.set_title("Sepal Length vs Petal Length")
ax.legend()

st.pyplot(fig)