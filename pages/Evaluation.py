import json
import wave
import streamlit as st

from io import BytesIO
import requests
import boto3
import librosa
import tensorflow as tf
import numpy as np

import pandas as pd
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())

st.set_page_config(layout="wide")


def soundx_tflite_prediction(waveform, tflite_model="/tmp/soundx_model.tflite"):
    # Download the model to yamnet-classification.tflite
    interpreter = tf.lite.Interpreter(tflite_model)

    input_details = interpreter.get_input_details()
    waveform_input_index = input_details[0]["index"]
    output_details = interpreter.get_output_details()
    scores_output_index = output_details[0]["index"]

    interpreter.resize_tensor_input(waveform_input_index, [waveform.size], strict=False)
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, waveform)
    interpreter.invoke()
    scores = interpreter.get_tensor(scores_output_index)

    sorted_scores_indices = np.argsort(scores)[::-1]
    return scores, sorted_scores_indices


AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


client = boto3.client(
    "s3",
    region_name="eu-west-3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=boto3.session.Config(signature_version="s3v4"),
)
s3 = boto3.resource(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)
model_bucket = s3.Bucket("soundx-models")
dataset_bucket = s3.Bucket("soundx-audio-dataset")

@st.cache_data(ttl=3600)
def load_models():
    models = list()
    for obj in model_bucket.objects.all():
        models.append(obj.key.split("/")[0])
    return sorted(list(set(models)))[::-1]

@st.cache_data(ttl=3600)
def load_folders():
    folders = list()
    for obj in dataset_bucket.objects.all():
        folders.append(obj.key.split("/")[0])
    return sorted(list(set(folders)))

@st.cache_data(ttl=3600)
def load_data(folder=":TESTS"):
    print(folder)
    files = list()
    for obj in dataset_bucket.objects.filter(Prefix=folder):
        if obj.key.endswith(".wav"):
            label, file = obj.key.split("/")[0], obj.key.split("/")[1]
            url = client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": "soundx-audio-dataset", "Key": f"{label}/{file}"},
                ExpiresIn=3600,
            )
            content = BytesIO(requests.get(url).content)
            signal_wave = wave.open(content, "r")
            frames = signal_wave.getnframes()
            rate = signal_wave.getframerate()
            samp_width = signal_wave.getsampwidth()
            n_channels = signal_wave.getnchannels()
            full_sig = np.frombuffer(signal_wave.readframes(frames), dtype=np.int16)
            # sig = full_sig[::n_channels]
            with wave.open(f"/tmp/{file}.wav", "w") as outfile:
                outfile.setnchannels(n_channels)
                outfile.setsampwidth(samp_width)
                outfile.setframerate(rate)
                outfile.setnframes(int(len(full_sig) / samp_width))
                outfile.writeframes(full_sig)
            files.append(file)
    return files

models = load_models()

model = st.selectbox("Select a model", models, index=models.index("latest"))

folders = load_folders()

folder = st.selectbox("Select a class", folders, index=folders.index(":TESTS"))

files = load_data(folder)

print(model)

classes = json.loads(
    s3.Object("soundx-models", f"{model}/soundx_model_general.json")
    .get()["Body"]
    .read()
    .decode("utf-8")
)["classes"]

tflite_model = client.download_file(
    Bucket="soundx-models",
    Key=f"{model}/soundx_model_general.tflite",
    Filename="/tmp/soundx_model_general.tflite",
)

all_specific_classes = list()
for model_class in classes:
    tflite_model = client.download_file(
        Bucket="soundx-models",
        Key=f"{model}/soundx_model_{model_class}.tflite",
        Filename=f"/tmp/soundx_model_{model_class}.tflite",
    )

    all_specific_classes.extend(json.loads(
        s3.Object("soundx-models", f"{model}/soundx_model_{model_class}.json")
        .get()["Body"]
        .read()
        .decode("utf-8")
    )["classes"])

target_sr = 16000


@st.cache_data(ttl=600, )
def generate_result_dataframe(model, files, target_label=None):
    df = pd.DataFrame(
        columns=["File", "Target Label", "General label", "Specific label", "Prediction certainty"]
    )
    for file in files:
        resampled_signal, sr = librosa.load(f"/tmp/{file}.wav", sr=target_sr)

        scores, scores_idx = soundx_tflite_prediction(
            resampled_signal, tflite_model="/tmp/soundx_model_general.tflite"
        )

        general_label = classes[scores_idx[0]]

        specific_classes = json.loads(
            s3.Object("soundx-models", f"{model}/soundx_model_{general_label}.json")
            .get()["Body"]
            .read()
            .decode("utf-8")
        )["classes"]

        specific_scores, specific_scores_idx = soundx_tflite_prediction(
            resampled_signal, tflite_model=f"/tmp/soundx_model_{general_label}.tflite"
        )

        specific_label = specific_classes[specific_scores_idx[0]]

        if file[:-4] in all_specific_classes:
            target_label = file[:-4]
        elif file[:-6] in all_specific_classes:
            target_label = file[:-6]

        if target_label == specific_label:
            certainty = (
                scores[scores_idx[0]] * specific_scores[specific_scores_idx[0]] * 100
            )
        elif target_label in specific_classes:
            tmp_classes = np.array(specific_classes)
            idx = np.where(tmp_classes == target_label)[0][0]
            certainty = (
                scores[scores_idx[0]] * specific_scores[idx] * 100
            )
        else:
            certainty = 0

        s = pd.DataFrame(
            [
                file,
                target_label,
                general_label,
                specific_label,
                certainty,
            ],
            index=df.columns,
        )
        df = pd.concat([df, s.T], axis=0)
    return df


df = generate_result_dataframe(model, files, target_label=folder)
st.write("Results for each file")
st.dataframe(df)

results = df[['Target Label', 'Prediction certainty']].groupby("Target Label").mean()
st.write("Average results for each class")
st.dataframe(results)
