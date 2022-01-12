import sys
import numpy as np
import pandas as pd
import cv2 
import tensorflow as tf
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Helpers to turn any kind of data into bytes
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_png(value).numpy()])
    )

# WRITE

def serialize_example_covnet(id, image, label, path):
    '''
    Function that turns one set (!) of id, image, label into bytes

    Returns:    Representation of set in bytes
    '''  
    image = tf.io.decode_png(tf.io.read_file(path+image))
    feature = {
        "id" : bytes_feature(id),
        'image' : image_feature(image),
        'label' : _float_feature(label), 
    }

    #  Create a Features message using tf.train.Example (some tensorflow thing)
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_features(ids, images, labels, path = "data/images_resized/", filename:str="images"):
    '''
    Function that writes bytes into a tfrec-file
    '''  
    filename= filename+".tfr"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0
    with tf.io.TFRecordWriter(filename) as writer:  
        for id, image, label in tqdm(zip(ids, images, labels), total = len(labels)):
            example = serialize_example_covnet(id, image, label, path)
            writer.write(example)
            count += 1

    return count


# READ

def make_dataset(file_path, batch_size, seed = 123):
  """Creates a `tf.data.TFRecordDataset`.

  Args:
    file_path: Name of the file in the `.tfrecord` format containing
      `tf.train.Example` objects.
    training: Boolean indicating if we are in training mode.

  Returns:
    An instance of `tf.data.TFRecordDataset` containing the `tf.train.Example`
    objects.
  """
  tf.random.set_seed(seed)
  def parse_example(example_proto):
    """Extracts relevant fields from the `example_proto`.

    Args:
      example_proto: An instance of `tf.train.Example`.

    Returns:
      A pair whose first value is a dictionary containing relevant features
      and whose second value contains the ground truth label.
    """
    # The 'words' feature is a multi-hot, bag-of-words representation of the
    # original raw text. A default value is required for examples that don't
    # have the feature.
    feature_spec = {
        #"id": tf.io.FixedLenFeature((), tf.string),
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.float32),
    }

    example = tf.io.parse_single_example(example_proto, feature_spec)
   # the image needs further treatment

    image = tf.io.decode_png(example['image'],channels =3)

    #image = tf.squeeze(image)
    example["input_1"] = image
    label = example.pop('label') 
    example.pop('image') 

    return example, label

  dataset = tf.data.TFRecordDataset([file_path])
  #if training:
   #dataset = dataset.shuffle(6000)
  dataset = dataset.map(parse_example)  
  dataset = dataset.batch(batch_size)
  return dataset