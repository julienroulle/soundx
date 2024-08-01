import datetime
import json
from pathlib import Path
from modal import App, CloudBucketMount, Image, Secret

s3_bucket_name = "soundx-audio-dataset"

DATA_PATH: Path = Path("/data")
MODEL_PATH: Path = Path("/models")
PRESET_PATH: Path = Path("/presets")

image = Image.from_registry("tensorflow/tensorflow:2.16.1-gpu").pip_install(
    "boto3",
    "tensorflow==2.16.1",
    "keras",
    "numpy==1.26.2",
    "pandas==2.1.3",
    "tensorflow_io",
    "tensorflow_hub",
    "scikit-learn",
    "tblib",
)
app = App(
    "soundx",
    image=image,
)

with image.imports():
    # from tensorflow.python.framework.ops import disable_eager_execution

    # disable_eager_execution()

    import numpy as np
    import pandas as pd

    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_io as tfio

    import keras

    from keras.optimizers import Adam
    from sklearn.model_selection import KFold


@app.function(
    image=image,
    volumes={
        DATA_PATH: CloudBucketMount(
            s3_bucket_name,
            secret=Secret.from_name("s3-bucket-secret"),
        ),
    },
)
def load_data():
    pd_data = pd.DataFrame()
    for f in DATA_PATH.rglob("*.wav"):
        filename = str(f)
        if filename.startswith("/data/:TEST"):
            continue

        label = filename.split("/")[2]
        category = label.split("_")[0]

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
    pd_data = pd_data.reset_index()
    return pd_data.to_json(orient="columns")


@app.function(
    image=image,
    volumes={
        DATA_PATH: CloudBucketMount(
            s3_bucket_name,
            secret=Secret.from_name("s3-bucket-secret"),
        ),
        MODEL_PATH: CloudBucketMount(
            "soundx-models",
            secret=Secret.from_name("s3-bucket-secret"),
        ),
        PRESET_PATH: CloudBucketMount(
            "soundx-presets",
            secret=Secret.from_name("s3-bucket-secret"),
        ),
    },
    # gpu="t4",
    timeout=60 * 60 * 24,
)
def train(pd_data, today, target="label", name="", validation_sets=None):
    path = MODEL_PATH / today

    pd_data = pd.read_json(pd_data, orient="columns")

    yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
    yamnet_model = hub.load(yamnet_model_handle)

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
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
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

    filenames = filtered_pd["filename"]
    targets = filtered_pd["target"]
    folds = filtered_pd["fold"]

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
                shape=(1024,), dtype=tf.float32, name="input_embedding"
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
        validation_data=val_ds,
        epochs=200,
        # epochs=1,
        callbacks=callback,
    )

    loss, model_accuracy = soundx_model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", model_accuracy)

    saved_model_path = f"soundx_model{name}.keras"

    input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name="audio")
    embedding_extraction_layer = hub.KerasLayer(
        yamnet_model_handle, trainable=False, name="yamnet"
    )

    # _, embeddings_output, _ = embedding_extraction_layer(input_segment)
    # serving_outputs = soundx_model(embeddings_output)

    @keras.saving.register_keras_serializable(package="MyLayers")
    class MyLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(MyLayer, self).__init__(**kwargs)

        def call(self, input):
            _, embeddings_output, _ = embedding_extraction_layer(input)
            serving_outputs = soundx_model(embeddings_output)
            serving_outputs = ReduceMeanLayer(axis=0, name="classifier")(
                serving_outputs
            )

            serving_outputs = tf.expand_dims(serving_outputs, axis=0)
            serving_outputs = tf.keras.layers.Softmax(axis=-1)(serving_outputs)
            serving_outputs = tf.squeeze(serving_outputs, axis=0)
            return serving_outputs

    serving_outputs = MyLayer()(input_segment)

    serving_model = tf.keras.Model(input_segment, serving_outputs)
    serving_model.save(saved_model_path)

    # reloaded_model = tf.saved_model.load(saved_model_path)
    reloaded_model = tf.keras.models.load_model(saved_model_path)

    tf_callable = tf.function(
        reloaded_model.call,
        autograph=False,
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)],
    )
    tf_concrete_function = tf_callable.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tf_concrete_function], tf_callable
    )

    # converter = tf.lite.TFLiteConverter.from_keras_model(reloaded_model)
    tflite_model = converter.convert()

    # Save the model.
    with open(path / f"soundx_model{name}.tflite", "wb") as f:
        f.write(tflite_model)

    with open(path / f"soundx_model{name}.json", "w") as f:
        json.dump({"classes": list(my_classes)}, f)

    with open(path / f"Classes{name}.h", "w") as f:
        f.write(f"// Number of presets: {len(my_classes)}\n")
        f.write(f"// This file was generated on {today}\n\n")

        f.write("#include <array>\n\n")
        f.write(
            f"std::array<const char*, {len(my_classes)}> labels{name.replace('-', '_').lower()} = "
        )
        f.write('{\n    "' + '",\n    "'.join([str(x) for x in my_classes]) + '"\n};')

    if name == "":
        with open(path / "Settings.cpp", "w") as f:
            f.write('#include "Settings.h"\n\n')
            f.write("/**\n")
            f.write(" * Constructor\n")
            f.write(" * @return void\n")
            f.write(" * @note: the presets values are hardcoded here\n")
            f.write(" **/\n")
            f.write("SettingsStruct::SettingsStruct()\n{\n")
            for file in PRESET_PATH.rglob("*.txt"):
                with open(file, "rb") as preset:
                    lines = preset.readlines()
                    f.write("\tPresets.insert({")
                    f.write('"{}", '.format(str(file)[:-4]))
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

    if name != "_general":
        validation_sets = pd.read_json(validation_sets, orient="columns")
        validation_sets = pd.concat(
            [validation_sets, filtered_pd.loc[filtered_pd["fold"] == 5]]
        )
        validation_sets = validation_sets.reset_index(drop=True)
        return validation_sets.to_json(orient="columns")
    else:
        filtered_pd.loc[filtered_pd["fold"] == 5].to_csv(
            str(MODEL_PATH / today / "validation_general.csv")
        )

    # s3.meta.client.upload_file(
    #     f"/tmp/soundx_model{name}.tflite",
    #     "soundx-models",
    #     f"{today}/soundx_model{name}.tflite",
    # )
    # s3.meta.client.upload_file(
    #     f"/tmp/soundx_model{name}.json",
    #     "soundx-models",
    #     f"{today}/soundx_model{name}.json",
    # )
    # s3.meta.client.upload_file(
    #     f"/tmp/Classes{name}.h", "soundx-models", f"{today}/Classes{name}.h"
    # )
    # s3.meta.client.upload_file(
    #     "/tmp/Settings.cpp", "soundx-models", f"{today}/Settings.cpp"
    # )

    # s3.meta.client.upload_file(
    #     f"/tmp/soundx_model{name}.tflite",
    #     "soundx-models",
    #     f"latest/soundx_model{name}.tflite",
    # )
    # s3.meta.client.upload_file(
    #     f"/tmp/soundx_model{name}.json",
    #     "soundx-models",
    #     f"latest/soundx_model{name}.json",
    # )
    # s3.meta.client.upload_file(
    #     f"/tmp/Classes{name}.h", "soundx-models", f"latest/Classes{name}.h"
    # )
    # s3.meta.client.upload_file(
    #     "/tmp/Settings.cpp", "soundx-models", "latest/Settings.cpp"
    # )


@app.function(
    timeout=60 * 60 * 24,
    image=image,
    volumes={
        MODEL_PATH: CloudBucketMount(
            "soundx-models",
            secret=Secret.from_name("s3-bucket-secret"),
        ),
    },
    # schedule=Cron("0 14 * * *"),
)
def main():
    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    path = MODEL_PATH / today
    path.mkdir(parents=True, exist_ok=True)

    pd_data = load_data.remote()

    validation_sets = pd.DataFrame().to_json(orient="columns")

    train.remote(pd_data, today, target="category", name="general")
    for group_name, group in pd.read_json(pd_data, orient="columns").groupby(
        "category"
    ):
        print(group_name, len(group))
        validation_sets = train.remote(
            group.to_json(orient="columns"),
            today,
            target="label",
            name=group_name,
            validation_sets=validation_sets,
        )

    pd.read_json(validation_sets, orient="columns").to_csv(
        str(path / "validation_sets.csv")
    )


@app.local_entrypoint()
def local_entrypoint():
    main.remote()
