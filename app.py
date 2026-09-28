import streamlit as st
import joblib

model = joblib.load("iris_model.pkl")

st.title("🌸 Iris Flower Prediction")

sl = st.number_input("Sepal Length (cm)", min_value=0.0)
sw = st.number_input("Sepal Width (cm)", min_value=0.0)
pl = st.number_input("Petal Length (cm)", min_value=0.0)
pw = st.number_input("Petal Width (cm)", min_value=0.0)

if st.button("Predict"):
    prediction = model.predict([[sl, sw, pl, pw]])
    st.success(f"Predicted Flower: {prediction[0].title()}")