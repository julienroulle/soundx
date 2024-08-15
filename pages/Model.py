from collections import defaultdict
import json
import wave
import streamlit as st

from io import BytesIO
import requests
import boto3
import librosa
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())

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


@st.cache_data(ttl=600)
def load_models():
    models = dict()
    for obj in model_bucket.objects.all():
        if obj.key.endswith(".tflite"):
            model_date = obj.key.split("/")[0]
            model_path = obj.key
            models[model_date] = model_path
    return sorted(list(set(models)))[::-1]


@st.cache_data(ttl=600)
def load_data():
    files = defaultdict(list)
    for obj in dataset_bucket.objects.all():
        if obj.key.endswith(".wav"):
            label, file = obj.key.split("/")[0], obj.key.split("/")[1]
            files[label].append(file)
    return files


files = load_data()
models = load_models()

model = st.selectbox("Select a model", models, index=models.index("latest"))

col1, col2 = st.columns([3, 2])

with col1:
    label = st.selectbox("Select label", files)
with col2:
    file = st.selectbox("Select file", files[label])

uploaded_file = st.file_uploader("Upload a file")
print(uploaded_file, uploaded_file is not None)
if uploaded_file is not None:
    signal_wave = wave.open(uploaded_file, "r")
else:
    url = client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": "soundx-audio-dataset", "Key": f"{label}/{file}"},
        ExpiresIn=3600,
    )
    content = BytesIO(requests.get(url).content)
    signal_wave = wave.open(content, "r")

try:
    frames = signal_wave.getnframes()
    rate = signal_wave.getframerate()
    samp_width = signal_wave.getsampwidth()
    n_channels = signal_wave.getnchannels()
    full_sig = np.frombuffer(signal_wave.readframes(frames), dtype=np.int16)
    sig = full_sig[::n_channels]

    with wave.open("/tmp/export.wav", "w") as outfile:
        outfile.setnchannels(n_channels)
        outfile.setsampwidth(samp_width)
        outfile.setframerate(rate)
        outfile.setnframes(int(len(full_sig) / samp_width))
        outfile.writeframes(full_sig)

    duration = frames / float(rate)
except wave.Error:
    sig = []

# audio_file = BytesIO(requests.get(url).content)

# # create waveform plot
# fig = plt.figure(figsize=(10, 3))
# ax = fig.add_subplot(111)
# ax.plot(sig)
# ax.set_xlim([0, len(sig)])
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('Amplitude')
# st.pyplot(fig)

# st.audio(audio_file.read(), format='audio/wav')

classes = json.loads(
    s3.Object("soundx-models", f"{model}/soundx_model.json")
    .get()["Body"]
    .read()
    .decode("utf-8")
)["classes"]


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


target_sr = 16000
resampled_signal, sr = librosa.load("/tmp/export.wav", sr=target_sr)

tflite_model = client.download_file(
    Bucket="soundx-models",
    Key=f"{model}/soundx_model.tflite",
    Filename="/tmp/soundx_model.tflite",
)
scores, scores_idx = soundx_tflite_prediction(resampled_signal)

predictions_columns = st.columns([3, 2, 3, 3])
predictions_columns[0].subheader("Prediction")

for idx in range(5):
    predictions_columns[0].metric(
        classes[scores_idx[idx]], f"{scores[scores_idx[idx]] * 100:.1f} %"
    )

predictions_columns[1].subheader("General Prediction")
tflite_model = client.download_file(
    Bucket="soundx-models",
    Key=f"{model}/soundx_model_general.tflite",
    Filename="/tmp/soundx_model_general.tflite",
)
classes = json.loads(
    s3.Object("soundx-models", f"{model}/soundx_model_general.json")
    .get()["Body"]
    .read()
    .decode("utf-8")
)["classes"]

scores, scores_idx = soundx_tflite_prediction(
    resampled_signal, tflite_model="/tmp/soundx_model_general.tflite"
)

for idx in range(5):
    predictions_columns[1].metric(
        classes[scores_idx[idx]], f"{scores[scores_idx[idx]] * 100:.1f} %"
    )

general_label, second_label = classes[scores_idx[0]], classes[scores_idx[1]]

predictions_columns[2].subheader("Specific Prediction 1st label")
tflite_model = client.download_file(
    Bucket="soundx-models",
    Key=f"{model}/soundx_model_{general_label}.tflite",
    Filename=f"/tmp/soundx_model_{general_label}.tflite",
)
classes = json.loads(
    s3.Object("soundx-models", f"{model}/soundx_model_{general_label}.json")
    .get()["Body"]
    .read()
    .decode("utf-8")
)["classes"]

scores, scores_idx = soundx_tflite_prediction(
    resampled_signal, tflite_model=f"/tmp/soundx_model_{general_label}.tflite"
)

for idx in range(5):
    predictions_columns[2].metric(
        classes[scores_idx[idx]], f"{scores[scores_idx[idx]] * 100:.1f} %"
    )

predictions_columns[3].subheader("Specific Prediction 2nd label")
tflite_model = client.download_file(
    Bucket="soundx-models",
    Key=f"{model}/soundx_model_{second_label}.tflite",
    Filename=f"/tmp/soundx_model_{second_label}.tflite",
)
classes = json.loads(
    s3.Object("soundx-models", f"{model}/soundx_model_{second_label}.json")
    .get()["Body"]
    .read()
    .decode("utf-8")
)["classes"]

scores, scores_idx = soundx_tflite_prediction(
    resampled_signal, tflite_model=f"/tmp/soundx_model_{second_label}.tflite"
)

for idx in range(5):
    predictions_columns[3].metric(
        classes[scores_idx[idx]], f"{scores[scores_idx[idx]] * 100:.1f} %"
    )
