import streamlit as st
import modal
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())
st.title("Train a model")


def train_model():
    predict_fn = modal.Function.lookup("soundx", "main")
    function_call = predict_fn.spawn()
    st.write("Training model...")


@st.dialog("Start a new training?")
def start_training():
    reason = st.text_input("Password")
    if st.button("Submit"):
        print(reason, os.environ.get("STREAMLIT_PASSWORD"))
        if reason == os.environ.get("STREAMLIT_PASSWORD"):
            st.session_state.training = True
            train_model()
        else:
            st.write("Incorrect password")


if "training" not in st.session_state:
    print(os.environ.get("STREAMLIT_PASSWORD"))
    start_training()
