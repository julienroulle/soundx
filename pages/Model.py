import json
import wave
import streamlit as st

from io import BytesIO
import requests
import boto3
import librosa
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dotenv import find_dotenv, load_dotenv
import os

from pages.Dataset import load_data
load_dotenv(find_dotenv())

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


client = boto3.client('s3', region_name='eu-west-3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, config=boto3.session.Config(signature_version='s3v4'))
s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,)
bucket = s3.Bucket("soundx-models")

@st.cache_data
def load_models():
    models = dict()
    for obj in bucket.objects.all():
        if obj.key.endswith('.tflite'):
            model_date = obj.key.split('/')[0]
            model_path = obj.key
            models[model_date] = model_path
    return models

files = load_data()
models = load_models()

model = st.selectbox('Select model', list(models.keys()), index=len(models.keys())-1)

col1, col2 = st.columns([3, 2])
print(col2)
with col1:
    label = st.selectbox("Select label", files)
with col2:
    file = st.selectbox("Select file", files[label])

url = client.generate_presigned_url(
        ClientMethod='get_object', 
        Params={'Bucket': 'soundx-audio-dataset', 'Key': f"{label}/{file}"},
        ExpiresIn=3600)
content = BytesIO(requests.get(url).content)

signal_wave = wave.open(content, 'r')
frames = signal_wave.getnframes()
rate = signal_wave.getframerate()
samp_width = signal_wave.getsampwidth()
n_channels = signal_wave.getnchannels()
full_sig = np.frombuffer(signal_wave.readframes(frames), dtype=np.int16)
sig = full_sig[::n_channels]

with wave.open('/tmp/export.wav', 'w') as outfile:
    outfile.setnchannels(n_channels)
    outfile.setsampwidth(samp_width)
    outfile.setframerate(rate)
    outfile.setnframes(int(len(full_sig) / samp_width))
    outfile.writeframes(full_sig)

duration = frames / float(rate)

# create waveform plot
fig = plt.figure(figsize=(10, 3))
ax = fig.add_subplot(111)
ax.plot(sig)
ax.set_xlim([0, len(sig)])
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
st.pyplot(fig)

audio_file = BytesIO(requests.get(url).content)
st.audio(audio_file.read(), format='audio/wav')

classes = json.loads(s3.Object("soundx-models", f"{model}/soundx_model.json").get()['Body'].read().decode('utf-8'))['classes']

def soundx_tflite_prediction(waveform, labels, tflite_model='/tmp/soundx_model.tflite'):
    # Download the model to yamnet-classification.tflite
    interpreter = tf.lite.Interpreter(tflite_model)

    input_details = interpreter.get_input_details()
    waveform_input_index = input_details[0]['index']
    output_details = interpreter.get_output_details()
    scores_output_index = output_details[0]['index']

    interpreter.resize_tensor_input(waveform_input_index, [waveform.size], strict=False)
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, waveform)
    interpreter.invoke()
    scores = interpreter.get_tensor(scores_output_index)

    top_class_index = scores.argmax()
    return labels[top_class_index]

sr = 44100
signal, sr = librosa.load('/tmp/export.wav', sr=sr)
target_sr = 16000

tflite_model = client.download_file(Bucket='soundx-models', Key=f'{model}/soundx_model.tflite', Filename='/tmp/soundx_model.tflite')
resampled_signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
predicted_class = soundx_tflite_prediction(resampled_signal, classes)

st.write(f"Prediction: {predicted_class}")