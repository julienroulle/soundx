import streamlit as st

import pandas as pd
from dotenv import find_dotenv, load_dotenv
import os
import boto3

load_dotenv(find_dotenv())

st.set_page_config(layout="wide")

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

s3 = boto3.resource(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)
model_bucket = s3.Bucket("soundx-models")


@st.cache_data(ttl=600)
def load_models():
    models = dict()
    for obj in model_bucket.objects.all():
        if obj.key.endswith(".tflite"):
            model_date = obj.key.split("/")[0]
            model_path = obj.key
            models[model_date] = model_path
    return sorted(list(set(models)))[::-1]


models = load_models()

model = st.selectbox("Select a model", models, index=models.index("latest"))

# for obj in model_bucket.objects.filter(Prefix=f"{model}/"):
#     print(obj.key)

results = pd.read_csv(
    s3.Object("soundx-models", f"{model}/validation_sets_results.csv").get()["Body"]
)

results_summary = pd.read_csv(
    s3.Object("soundx-models", f"{model}/validation_set_results_summary.csv").get()[
        "Body"
    ]
).sort_values("Prediction certainty", ascending=False)

st.header("Results")
st.dataframe(results, use_container_width=True)

st.header("Results summary")
st.dataframe(results_summary, use_container_width=True)
