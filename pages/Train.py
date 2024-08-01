import streamlit as st
import modal

st.title("Train a model")


def train_model():
    predict_fn = modal.Function.lookup("soundx", "main")
    function_call = predict_fn.spawn()
    st.write("Training model...")


st.button("Run", on_click=train_model)
