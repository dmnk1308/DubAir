import sys
import numpy as np
import pandas as pd
import cv2 
import tensorflow as tf
from tensorflow import keras
import itertools
import io
from sklearn import metrics
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt

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

# Write tfr files
def serialize_example(image, label, path):
    '''
    Function that turns one set (!) of id, image, label into bytes

    Returns:    Representation of set in bytes
    '''  
    image = tf.io.decode_png(tf.io.read_file(path+image))
    feature = {
        #"id" : bytes_feature(id),
        'image' : image_feature(image),
        'label' : _int64_feature(label), 
    }

    #  Create a Features message using tf.train.Example (some tensorflow thing)
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# for images without labels/descriptions
def serialize_example_pred(image, path):
    '''
    Function that turns one set (!) of id, image, label into bytes

    Returns:    Representation of set in bytes
    '''  
    image = tf.io.decode_png(tf.io.read_file(path+image))
    feature = {
        #"id" : bytes_feature(id),
        'image' : image_feature(image),
    }

    #  Create a Features message using tf.train.Example (some tensorflow thing)
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_features(images, labels = None, path = "/Users/dmnk/Downloads/images_resized/", filename:str="images", prediction = False):
    '''
    Function that writes bytes into a tfrec-file
    '''  
    filename= filename+".tfr"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0
    with tf.io.TFRecordWriter(filename) as writer:  
#        for id, image, label in zip(ids, images, labels):
      
      if prediction:
          for image in tqdm(images):
              try:
                  example = serialize_example_pred(image, path)
                  writer.write(example)
                  count += 1
              except:
                  continue
          return count

      for image, label in tqdm(zip(images, labels), total = len(labels)):
          try:
              example = serialize_example(image, label, path)
              writer.write(example)
              count += 1
          except:
            continue

    return count


# Read tfr files and form a dataset from them
def make_dataset(file_path, batch_size, seed = 123, prediction = False, training = False):
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
  def parse_example(example_proto, prediction):
    """Extracts relevant fields from the `example_proto`.

    Args:
      example_proto: An instance of `tf.train.Example`.

    Returns:
      A pair whose first value is a dictionary containing relevant features
      and whose second value contains the ground truth label.
    """

    if prediction:
        feature_spec = {
        #"id": tf.io.FixedLenFeature((), tf.string),
        'image': tf.io.FixedLenFeature((), tf.string),
        }

        example = tf.io.parse_single_example(example_proto, feature_spec)
    
        image = tf.io.decode_png(example['image'],channels =3)

        #image = tf.squeeze(image)
        example["input_1"] = image
        #id = example.pop("id") 
        example.pop('image') 

        return example


    feature_spec = {
        #"id": tf.io.FixedLenFeature((), tf.string),
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_spec)
    
   # the image needs further treatment
    image = tf.io.decode_png(example['image'],channels =3)
    example["input_1"] = image
    label = example.pop('label')
    #id = example.pop("id") 
    example.pop('image') 

    return example, label

  dataset = tf.data.TFRecordDataset([file_path])
  if training:
       dataset = dataset.shuffle(6000)
  dataset = dataset.map(lambda x: parse_example(x, prediction=prediction))
  dataset = dataset.batch(batch_size)
  return dataset

# function to plot some pictures and their description
def show_category(cat, df, n = 10, folder = "/Users/dmnk/Downloads/images_resized/"):
  ''' Show pictures and their description for a category '''
  filter = df[cat] == 1
  path = folder + df["img_path"][filter][:n]
  l = df["label_raw"][filter][:n]
  id = df["img_path"][filter][:n]

  #path = random.choices(list(path), k=n)

  for p, lab, i in zip(path, l, id):
      img = plt.imread(str(p))
      print(i) 
      print(lab)
      plt.imshow(img)
      plt.show()



######################## MODEL Definition ############################

def train_model(traindata, valdata, lr, epochs, logdir,  label_book, weights, callbacks = [], l2_reg = 0, do = 0):
  ''' Setup model and train it '''
  # Define Input
  inputs = keras.layers.Input(shape = (None, None, 3), name = "input_1")
  x = tf.cast(inputs, tf.float32)
  x = keras.applications.resnet50.preprocess_input(x)

  # Load ResNet with pretrained Imagenet weights
  resnet = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', pooling="avg")

  # freeze the weights
  resnet.trainable = False
  x = resnet(x)

  # Layer 2 - Dense ReLu
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dropout(do)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Dense(512, activation = "relu", kernel_regularizer = keras.regularizers.l2(l2_reg))(x)

  # Layer 3 - Dense ReLu 
  x = keras.layers.Dropout(do)(x)
  x = keras.layers.BatchNormalization()(x)
  outputs = keras.layers.Dense(7)(x)

  # Combine pretrained and output model
  model = keras.Model(inputs, outputs)

  tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
  callbacks.append(tensorboard_callback)

  model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

  history = model.fit(traindata, validation_data = valdata, epochs = epochs, callbacks = callbacks, class_weight = weights)
  return model, history