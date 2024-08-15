from io import BytesIO
import wave
import librosa
import pandas as pd
import requests
import tensorflow as tf
import numpy as np
import os
import boto3
import json

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


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

classes = json.loads(
    s3.Object("soundx-models", "latest/soundx_model_general.json")
    .get()["Body"]
    .read()
    .decode("utf-8")
)["classes"]

tflite_model = client.download_file(
    Bucket="soundx-models",
    Key="latest/soundx_model_general.tflite",
    Filename="/tmp/soundx_model_general.tflite",
)

all_specific_classes = list()
for model_class in classes:
    tflite_model = client.download_file(
        Bucket="soundx-models",
        Key=f"latest/soundx_model_{model_class}.tflite",
        Filename=f"/tmp/soundx_model_{model_class}.tflite",
    )

    all_specific_classes.extend(
        json.loads(
            s3.Object("soundx-models", f"latest/soundx_model_{model_class}.json")
            .get()["Body"]
            .read()
            .decode("utf-8")
        )["classes"]
    )

target_sr = 16000


def generate_general_dataframe(files, target_label=None):
    df = pd.DataFrame(
        columns=[
            "File",
            "Target Label",
            "General label",
            "Prediction certainty",
        ]
    )
    for file in files:
        resampled_signal, sr = librosa.load(file, sr=target_sr)

        scores, scores_idx = soundx_tflite_prediction(
            resampled_signal, tflite_model="/tmp/soundx_model_general.tflite"
        )

        general_label = classes[scores_idx[0]]

        target_label = file.split("/")[-2].split("_")[0]

        print(target_label, general_label)

        if target_label == general_label:
            certainty = scores[scores_idx[0]] * 100
        elif target_label in classes:
            tmp_classes = np.array(classes)
            idx = np.where(tmp_classes == target_label)[0][0]
            certainty = scores[idx] * 100
        else:
            certainty = 0

        s = pd.DataFrame(
            [
                file,
                target_label,
                general_label,
                certainty,
            ],
            index=df.columns,
        )
        df = pd.concat([df, s.T], axis=0)
    return df


def generate_result_dataframe(files, target_label=None):
    df = pd.DataFrame(
        columns=[
            "File",
            "Target Label",
            "General label",
            "Specific label",
            "Prediction certainty",
        ]
    )
    for file in files:
        resampled_signal, sr = librosa.load(file, sr=target_sr)

        scores, scores_idx = soundx_tflite_prediction(
            resampled_signal, tflite_model="/tmp/soundx_model_general.tflite"
        )

        general_label = classes[scores_idx[0]]

        specific_classes = json.loads(
            s3.Object("soundx-models", f"latest/soundx_model_{general_label}.json")
            .get()["Body"]
            .read()
            .decode("utf-8")
        )["classes"]

        specific_scores, specific_scores_idx = soundx_tflite_prediction(
            resampled_signal, tflite_model=f"/tmp/soundx_model_{general_label}.tflite"
        )

        specific_label = specific_classes[specific_scores_idx[0]]

        target_label = file.split("/")[-2]

        print(target_label, general_label, specific_label)

        if target_label == specific_label:
            certainty = (
                scores[scores_idx[0]] * specific_scores[specific_scores_idx[0]] * 100
            )
        elif target_label in specific_classes:
            tmp_classes = np.array(specific_classes)
            idx = np.where(tmp_classes == target_label)[0][0]
            certainty = scores[scores_idx[0]] * specific_scores[idx] * 100
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


files = pd.read_csv("data/processed/validation_general.csv")["filename"].tolist()

df = generate_general_dataframe(files)

df.to_csv("data/processed/validation_general_results.csv", index=False)

results = df[["Target Label", "Prediction certainty"]].groupby("Target Label").mean()

results.to_csv("data/processed/validation_general_results_summary.csv", index=True)

files = pd.read_csv("data/processed/validation_sets.csv")["filename"].tolist()

df = generate_result_dataframe(files)

df.to_csv("data/processed/validation_set_results.csv", index=False)

results = df[["Target Label", "Prediction certainty"]].groupby("Target Label").mean()

results.to_csv("data/processed/validation_set_results_summary.csv", index=True)
