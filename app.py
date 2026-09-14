
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

model = joblib.load("iris_model.pkl")

st.title("Iris Flower Prediction")

sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

if st.button("Predict"):
    pred = model.predict([[sl, sw, pl, pw]])

    flowers = ["Setosa", "Versicolor", "Virginica"]

    st.success("Prediction: " + flowers[pred[0]])

st.subheader("Iris Flower Analytics")

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=[
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width"
    ]
)

df["Flower"] = iris.target_names[iris.target]

fig, ax = plt.subplots()

for flower in df["Flower"].unique():
    data = df[df["Flower"] == flower]
    ax.scatter(
        data["Petal Length"],
        data["Petal Width"],
        label=flower
    )

ax.set_xlabel("Petal Length")
ax.set_ylabel("Petal Width")
ax.set_title("Iris Flower Scatter Plot")
ax.legend()

st.pyplot(fig)
