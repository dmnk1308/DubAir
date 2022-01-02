import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.applications.efficientnet import EfficientNetB0
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(image, image_name):
    feature = {"image_name": _bytes_feature(image_name), "image": _bytes_feature(image)}

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()




def tf_serialize_example(image, image_name):
    tf_string = tf.py_function(serialize_example, (image, image_name), tf.string)
    return tf.reshape(tf_string, ())


def process_path(file_path):
    parts = tf.strings.split(file_path, "/")
    image_name = tf.strings.split(parts[-1], ".")[0]
    raw = tf.io.read_file(file_path)
    return raw, image_name


def read_image_file_write_tfrecord(files_ds, output_filename):
    image_ds = files_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    serialized_features_dataset = image_ds.map(tf_serialize_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    writer = tf.data.experimental.TFRecordWriter(output_filename)
    writer.write(serialized_features_dataset)


def image_files_to_tfrecords(list_ds, output_folder, num_shard):
    start = time.time()
    for shard_id in range(0, num_shard):
        shard_list = list_ds.shard(num_shards=num_shard, index=shard_id)
        read_image_file_write_tfrecord(shard_list, output_folder + "/part-" + "{:03d}".format(shard_id) + ".tfrecord")
        print("Shard " + str(shard_id) + " saved after " + str(int(time.time() - start)) + "s")


feature_description = {
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
}


def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


def preprocess_image(d):
    image_name = d["image_name"]
    raw = d["image"]
    image = tf.image.decode_jpeg(raw)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, image_name


def read_tfrecord(filename):
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    return (
        raw_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .apply(tf.data.experimental.ignore_errors())
    )


def tfrecords_to_write_embeddings(tfrecords_folder, output_folder, model, batch_size):
    tfrecords = [
        f.numpy().decode("utf-8") for f in tf.data.Dataset.list_files(tfrecords_folder + "/*.tfrecord", shuffle=False)
    ]
    start = time.time()
    for shard_id, tfrecord in enumerate(tfrecords):
        shard = read_tfrecord(tfrecord)
        embeddings = images_to_embeddings(model, shard, batch_size)
        print("")
        print("Shard " + str(shard_id) + " done after " + str(int(time.time() - start)) + "s")
        save_embeddings_ds_to_parquet(
            embeddings, shard, output_folder + "/part-" + "{:03d}".format(shard_id) + ".parquet"
        )
        print("Shard " + str(shard_id) + " saved after " + str(int(time.time() - start)) + "s")


def list_files(images_path):
    return tf.data.Dataset.list_files(images_path + "/*", shuffle=False).cache()


def process_path(file_path):
    parts = tf.strings.split(file_path, "/")
    image_name = tf.strings.split(parts[-1], ".")[0]
    raw = tf.io.read_file(file_path)
    return raw, image_name


def read_data_from_files(list_ds):
    return list_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )  # .apply(tf.data.experimental.ignore_errors())


def images_to_embeddings(model, dataset, batch_size):
    return model.predict(dataset.batch(batch_size).map(lambda image_raw, image_name: image_raw), verbose=1)


def save_embeddings_ds_to_parquet(embeddings, dataset, path):
    embeddings = pa.array(embeddings.tolist(), type=pa.list_(pa.float32()))
    image_names = pa.array(dataset.map(lambda image_raw, image_name: image_name).as_numpy_iterator())
    table = pa.Table.from_arrays([image_names, embeddings], ["image_name", "embedding"])
    pq.write_table(table, path)


def compute_save_embeddings(list_ds, folder, num_shards, model, batch_size):
    start = time.time()
    for shard_id in range(0, num_shards):
        shard_list = list_ds.shard(num_shards=num_shards, index=shard_id)
        shard = read_data_from_files(shard_list)
        embeddings = images_to_embeddings(model, shard, batch_size)
        print("Shard " + str(shard_id) + " done after " + str(int(time.time() - start)) + "s")
        save_embeddings_ds_to_parquet(embeddings, shard, folder + "/part-" + "{:03d}".format(shard_id) + ".parquet")
        print("Shard " + str(shard_id) + " saved after " + str(int(time.time() - start)) + "s")
    print("Total time : " + str(int(time.time() - start)))


def run_inference_from_files(image_folder, output_folder, num_shards=10, batch_size=1000):
    model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    list_ds = list_files(image_folder)
    compute_save_embeddings(list_ds, output_folder, num_shards, model, batch_size)


def write_tfrecord(image_folder, output_folder, num_shards=10):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    list_ds = list_files(image_folder)
    image_files_to_tfrecords(list_ds, output_folder, num_shards)


def run_inference(tfrecords_folder, output_folder, batch_size=1000):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    tfrecords_to_write_embeddings(tfrecords_folder, output_folder, model, batch_size)


   

from dataclasses import dataclass
from IPython.display import Image, display
from ipywidgets import widgets, HBox, VBox
#import faiss
import random
import json


def read_embeddings(path):
    emb = pq.read_table(path).to_pandas()
    id_to_name = {k: v.decode("utf-8") for k, v in enumerate(list(emb["image_name"]))}
    name_to_id = {v: k for k, v in id_to_name.items()}
    embgood = np.stack(emb["embedding"].to_numpy())
    return [id_to_name, name_to_id, embgood]


def embeddings_to_numpy(input_path, output_path):
    emb = pq.read_table(input_path).to_pandas()

    Path(output_path).mkdir(parents=True, exist_ok=True)
    id_name = [{"id": k, "name": v.decode("utf-8")} for k, v in enumerate(list(emb["image_name"]))]
    json.dump(id_name, open(output_path + "/id_name.json", "w"))

    emb = np.stack(emb["embedding"].to_numpy())
    np.save(open(output_path + "/embedding.npy", "wb"), emb)


def build_index(emb):
    d = emb.shape[1]
    xb = emb
    index = faiss.IndexFlatIP(d)
    index.add(xb)
    return index


def random_search(path):
    [id_to_name, name_to_id, embeddings] = read_embeddings(path)
    index = build_index(embeddings)
    p = random.randint(0, len(id_to_name) - 1)
    print(id_to_name[p])
    results = search(index, id_to_name, embeddings[p])
    for e in results:
        print(f"{e[0]:.2f} {e[1]}")


def search(index, id_to_name, emb, k=5):
    D, I = index.search(np.expand_dims(emb, 0), k)  # actual search
    return list(zip(D[0], [id_to_name[x] for x in I[0]]))


def display_picture(image_path, image_name):
    display(Image(filename=f"{image_path}/{image_name}.jpeg"))


def display_results(image_path, results):
    hbox = HBox(
        [
            VBox(
                [
                    widgets.Label(f"{distance:.2f} {image_name}"),
                    widgets.Image(value=open(f"{image_path}/{image_name}.jpeg", "rb").read()),
                ]
            )
            for distance, image_name in results
        ]
    )
    display(hbox)