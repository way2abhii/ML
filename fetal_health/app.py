import streamlit as st
import numpy as np
import pickle

lr = pickle.load(open("lr_model.pkl", "rb"))
rf = pickle.load(open("rf_model.pkl", "rb"))
svm = pickle.load(open("svm_model.pkl", "rb"))

features = pickle.load(open("features.pkl", "rb"))

st.set_page_config(page_title="Fetal Health Predictor", layout="centered")

st.title("👶 Fetal Health Classification")
st.markdown("Fill all feature values below")

model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "SVM"])

st.subheader("Input Features")

input_values = []

cols = st.columns(3)

for i, feature in enumerate(features):
    with cols[i % 3]:
        val = st.number_input(feature, value=0.0)
        input_values.append(val)

if st.button("Predict"):

    input_array = np.array([input_values])

    if model_choice == "Logistic Regression":
        model = lr
        accuracy = "≈ 0.85"
    elif model_choice == "Random Forest":
        model = rf
        accuracy = "≈ 0.95"
    else:
        model = svm
        accuracy = "≈ 0.90"

    prediction = model.predict(input_array)[0]

    st.subheader("Result")

    if prediction == 1:
        st.success("Normal")
    elif prediction == 2:
        st.warning("Suspect")
    else:
        st.error("Pathological")

    st.info(f"Model Accuracy: {accuracy}")

    st.write("Input Shape:", input_array.shape)