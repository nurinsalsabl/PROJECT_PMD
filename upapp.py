import streamlit as st
import pickle

st.title("Load Model PKL")

uploaded_model = st.file_uploader("Upload your model (.pkl) file here", type=["pkl"])

if uploaded_model is not None:
    # Simpan file ke lokal
    with open("model.pkl", "wb") as f:
        f.write(uploaded_model.read())
    st.success("Model uploaded successfully!")

    # Load model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    st.write("Model loaded successfully!")
    # Contoh: pakai model.predict dll.
else:
    st.info("Please upload a .pkl file to continue.")
