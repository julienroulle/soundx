import datetime
import json
import os
import shutil

import re

import boto3
import librosa
import pandas as pd
import soundfile as sf
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight


today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
        print(logs)


class ReduceMeanLayer(tf.keras.layers.Layer):
    def __init__(self, axis=0, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, input):
        return tf.math.reduce_mean(input, axis=self.axis)


# Function to extract the embedding from the YAMNet model
@tf.function
def load_wav_16k_mono(filename):
    """Load a WAV file, convert it to a float tensor, resample to 16 kHz"""
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    return wav


def load_wav_for_map(filename, label, fold):
    return load_wav_16k_mono(filename), label, fold


# applies the embedding extraction model to a wav data
def extract_embedding(wav_data, label, fold):
    """run YAMNet to extract embedding from the wav data"""
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    num_embeddings = tf.shape(embeddings)[0]
    return (
        embeddings,
        tf.repeat(label, num_embeddings),
        tf.repeat(fold, num_embeddings),
    )


# Initialize the S3 client
boto3.setup_default_session(profile_name="soundx")
s3 = boto3.resource("s3")

# Specify the S3 bucket name and the local directory where you want to sync
# the data
s3_bucket = "soundx-audio-dataset"
local_directory = "data/raw"

# Load the YAMNet model
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"


def extract_path_from_error(error_string):
    # Define the pattern to match the path
    pattern = r"Trying to load a model of incompatible/unknown type. '(.*)' contains neither 'saved_model.pb' nor 'saved_model.pbtxt'."

    # Use re.search to find the path in the error string
    match = re.search(pattern, error_string)

    # If a match is found, return the path
    if match:
        return match.group(1)
    else:
        return None


try:
    yamnet_model = hub.load(yamnet_model_handle)
except ValueError as err:
    path = extract_path_from_error(str(err))
    shutil.rmtree(path)
    yamnet_model = hub.load(yamnet_model_handle)


validation_sets = pd.DataFrame()


def resample():
    shutil.rmtree("data/interim")
    for root, dirs, files in os.walk(local_directory, topdown=False):
        for f in files:
            if f.endswith(".wav"):
                print(f)
                new_dir = f'{root.replace("raw", "interim")}'

                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)

                signal, sr = librosa.load(f"{root}/{f}", sr=44100)

                target_sr = 16000

                if len(signal) == 0:
                    continue
                resampled_signal = librosa.resample(
                    signal, orig_sr=sr, target_sr=target_sr
                )

                resampled_file = f"{new_dir}/{f}"
                sf.write(resampled_file, resampled_signal, target_sr)


def train(pd_data, target="label", name=""):
    if name != "":
        name = f"_{name}"

    labels = pd_data[target].unique()
    targets_map = {label: idx for idx, label in enumerate(labels)}
    pd_data["target"] = pd_data[target].map(targets_map)
    NB_TRAINING_DATA = pd_data.groupby("target").count().min().min()
    print(f"Number of training data: {NB_TRAINING_DATA}")
    training_data = pd.DataFrame(columns=pd_data.columns)
    for idx, elt in pd_data.groupby("target"):
        if len(elt) >= NB_TRAINING_DATA:
            elt = elt.sample(NB_TRAINING_DATA)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            iter_ = kf.split(range(NB_TRAINING_DATA))
            train = np.concatenate(
                [next(iter_, None)[1], next(iter_, None)[1], next(iter_, None)[1]]
            )
            test = next(iter_, None)[1]
            val = next(iter_, None)[1]
            selected_data = elt.reset_index().drop(columns=["index"])
            selected_data.loc[train, "fold"] = 1
            selected_data.loc[test, "fold"] = 4
            selected_data.loc[val, "fold"] = 5
            training_data = pd.concat([training_data, selected_data])

    pd_data = training_data.copy()
    my_classes = pd_data[target].unique()
    map_class_to_id = {label: idx for idx, label in enumerate(my_classes)}

    filtered_pd = pd_data[pd_data[target].isin(my_classes)]
    class_id = filtered_pd[target].apply(lambda name: map_class_to_id[name])
    filtered_pd = filtered_pd.assign(target=class_id)

    full_path = filtered_pd["filename"].apply(
        lambda row: os.path.join(dest_data_path, row)
    )
    filtered_pd = filtered_pd.assign(filename=full_path)

    if name != "_general":
        global validation_sets
        validation_sets = pd.concat(
            [validation_sets, filtered_pd.loc[filtered_pd["fold"] == 5]]
        )
    else:
        filtered_pd.loc[filtered_pd["fold"] == 5].to_csv(
            "data/processed/validation_general.csv"
        )

    filenames = filtered_pd["filename"]
    targets = filtered_pd["target"]
    folds = filtered_pd["fold"]

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(targets), y=targets
    )

    main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))

    # extract embedding
    main_ds = main_ds.map(load_wav_for_map)
    main_ds = main_ds.map(extract_embedding).unbatch()

    cached_ds = main_ds.cache()
    train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 4)
    val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 4)
    test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

    # remove the folds column now that it's not needed anymore
    def remove_fold_column(embedding, label, fold):
        return embedding, label

    train_ds = train_ds.map(remove_fold_column)
    val_ds = val_ds.map(remove_fold_column)
    test_ds = test_ds.map(remove_fold_column)

    train_ds = train_ds.cache().shuffle(1000).batch(256).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(256).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().batch(256).prefetch(tf.data.AUTOTUNE)

    soundx_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(1024), dtype=tf.float32, name="input_embedding"
            ),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(len(my_classes)),
        ],
        name=f"soundx_model{name}",
    )

    soundx_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(learning_rate=2e-4),
        metrics=["accuracy"],
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=3, restore_best_weights=True
    )

    soundx_model.fit(
        train_ds,
        epochs=200,
        validation_data=val_ds,
        callbacks=callback,
    )

    loss, model_accuracy = soundx_model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", model_accuracy)

    saved_model_path = f"../models/soundx_model{name}"

    input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name="audio")
    embedding_extraction_layer = hub.KerasLayer(
        yamnet_model_handle, trainable=False, name="yamnet"
    )
    _, embeddings_output, _ = embedding_extraction_layer(input_segment)
    serving_outputs = soundx_model(embeddings_output)
    serving_outputs = ReduceMeanLayer(axis=0, name="classifier")(serving_outputs)

    serving_outputs = tf.expand_dims(serving_outputs, axis=0)
    serving_outputs = tf.keras.layers.Softmax(axis=-1)(serving_outputs)
    serving_outputs = tf.squeeze(serving_outputs, axis=0)

    serving_model = tf.keras.Model(input_segment, serving_outputs)
    serving_model.save(saved_model_path, include_optimizer=False)

    reloaded_model = tf.saved_model.load(saved_model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(reloaded_model)
    tflite_model = converter.convert()

    # Save the model.
    with open(f"/tmp/soundx_model{name}.tflite", "wb") as f:
        f.write(tflite_model)

    with open(f"/tmp/soundx_model{name}.json", "w") as f:
        json.dump({"classes": list(my_classes)}, f)

    with open(f"/tmp/Classes{name}.h", "w") as f:
        f.write(f"// Number of presets: {len(my_classes)}\n")
        f.write(f"// This file was generated on {today}\n\n")

        f.write("#include <array>\n\n")
        f.write(
            f"std::array<const char*, {len(my_classes)}> labels{name.replace('-', '_').lower()} = "
        )
        f.write('{\n    "' + '",\n    "'.join([str(x) for x in my_classes]) + '"\n};')

    with open("/tmp/Settings.cpp", "w") as f:
        f.write('#include "Settings.h"\n\n')
        f.write("/**\n")
        f.write(" * Constructor\n")
        f.write(" * @return void\n")
        f.write(" * @note: the presets values are hardcoded here\n")
        f.write(" **/\n")
        f.write("SettingsStruct::SettingsStruct()\n{\n")
        for obj in s3.Bucket("soundx-presets").objects.all():
            if obj.key == ".DS_Store":
                continue
            lines = obj.get()["Body"].readlines()
            f.write("\tPresets.insert({")
            f.write('"{}", '.format(obj.key[:-4]))
            f.write("PresetSettings(\n")
            line_number = len(lines)
            for idx, line in enumerate(lines):
                line = line.decode("utf8")
                line = line.replace("-inf", "-50")
                # split the line by whitespace
                settings_value = line.split()
                if settings_value[0].startswith("gain"):
                    f.write("\t\t{")
                    f.write("{}".format(", ".join(settings_value[1::])))
                    f.write("}")
                    if idx < line_number - 1:
                        f.write(",")
                    f.write(" // {}\n".format(settings_value[0]))
            f.write("\t)});\n\n")
        f.write("\tallClasses = new char*[Presets.size() + 1];\n")
        f.write("\tint i =0;\n")
        f.write(
            "\tfor (std::map<std::string, PresetSettings>::iterator it = Presets.begin(); it != Presets.end(); ++it) {\n"
        )
        f.write("\t\tallClasses[i] = new char[it->first.length() + 1];\n")
        f.write("\t\tstrcpy(allClasses[i], it->first.c_str());\n")
        f.write("\t\ti++;\n")
        f.write("\t}\n")
        f.write("}\n")
        f.write("SettingsStruct::~SettingsStruct()\n\n")
        f.write("{\n")
        f.write("\tfor (int i = 0; i < Presets.size(); i++) {\n")
        f.write("\t\tdelete[] allClasses[i];\n")
        f.write("\t}\n")
        f.write("\tdelete[] allClasses;\n")
        f.write("}\n")

    s3.meta.client.upload_file(
        f"/tmp/soundx_model{name}.tflite",
        "soundx-models",
        f"{today}/soundx_model{name}.tflite",
    )
    s3.meta.client.upload_file(
        f"/tmp/soundx_model{name}.json",
        "soundx-models",
        f"{today}/soundx_model{name}.json",
    )
    s3.meta.client.upload_file(
        f"/tmp/Classes{name}.h", "soundx-models", f"{today}/Classes{name}.h"
    )
    s3.meta.client.upload_file(
        "/tmp/Settings.cpp", "soundx-models", f"{today}/Settings.cpp"
    )

    s3.meta.client.upload_file(
        f"/tmp/soundx_model{name}.tflite",
        "soundx-models",
        f"latest/soundx_model{name}.tflite",
    )
    s3.meta.client.upload_file(
        f"/tmp/soundx_model{name}.json",
        "soundx-models",
        f"latest/soundx_model{name}.json",
    )
    s3.meta.client.upload_file(
        f"/tmp/Classes{name}.h", "soundx-models", f"latest/Classes{name}.h"
    )
    s3.meta.client.upload_file(
        "/tmp/Settings.cpp", "soundx-models", "latest/Settings.cpp"
    )


if __name__ == "__main__":
    resample()
    pd_data = pd.DataFrame(columns=["filename", "category", "label"])
    dest_data_path = "./data/interim"

    pd_data = pd.DataFrame()
    for d in os.listdir(dest_data_path):
        if d.endswith((".DS_Store", ".json", ".gitkeep")) or d.startswith(":TEST"):
            continue
        for f in os.listdir(f"{dest_data_path}/{d}"):
            label = d
            category = d.split("_")[0]
            filename = f"{d}/{f}"

            pd_data = pd.concat(
                [
                    pd_data,
                    pd.DataFrame(
                        [[filename, category, label]],
                        columns=["filename", "category", "label"],
                    ),
                ],
                axis=0,
            )

    train(pd_data, target="category", name="general")
    for group_name, group in pd_data.groupby("category"):
        print(group_name, len(group))
        train(group, target="label", name=group_name)

    validation_sets.to_csv("data/processed/validation_sets.csv")
