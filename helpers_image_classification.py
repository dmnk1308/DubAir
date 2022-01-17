import sys
import numpy as np
import pandas as pd
import cv2 
import tensorflow as tf
from tensorflow import keras
from keras.constraints import maxnorm
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

# WRITE

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

def write_features(images, labels = None, path = "data/images_resized/", filename:str="images", prediction = False):
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


# READ

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

def show_category(cat, df, n = 10, folder = "data/images_resized/"):
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



######################## MODEL ############################

def cm_cb(model, val_dataset, label_book, logdir):
  file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

  def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
    test_images = np.concatenate([x["input_1"] for x, y in val_dataset], axis=0)
    test_labels = np.concatenate([y for x, y in val_dataset], axis=0)

    test_pred_raw = model.predict(test_images)
    test_pred = np.argmax(test_pred_raw, axis=1)

    # Calculate the confusion matrix.
    cm = metrics.confusion_matrix(test_labels, test_pred)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=label_book["label"])
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


  def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

  def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      color = "white" if cm[i, j] > threshold else "black"
      plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
  cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
  return cm_callback

def train_model(traindata, valdata, lr, epochs, logdir,  label_book, weights, callbacks = []):
   
    # Define Input
    inputs = tf.keras.layers.Input(shape = (256, 256, 3), name = "input_1")

    # Load ResNet with pretrained Imagenet weights
    resnet = tf.keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', input_tensor=inputs, pooling="avg")

    # freeze the weights
    resnet.trainable = False
    outputs = keras.layers.BatchNormalization()(resnet.output)

    # Layer 1 - Flatten
    outputs = tf.keras.layers.Flatten()(outputs)
    #outputs = keras.layers.Dropout(.2)(outputs)
    # Layer 2 - Dense ReLu
    outputs = tf.keras.layers.Dense(1500, activation = "relu")(outputs)#, kernel_constraint=maxnorm(4))(outputs)
    # Layer 3 - Dense ReLu 
    #outputs = keras.layers.Dropout(.2)(outputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(500, activation = "relu")(outputs)#, kernel_constraint=maxnorm(4))(outputs)
    # Layer 4 - Dense Output 
    #outputs = keras.layers.Dropout(.2)(outputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(6)(outputs)

    # Combine pretrained and output model
    model = tf.keras.Model(inputs, outputs)

    cm_callback = cm_cb(model, val_dataset = valdata, logdir = logdir, label_book = label_book)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    callbacks.append(tensorboard_callback)
    callbacks.append(cm_callback)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr = lr),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])

    model.fit(traindata, validation_data = valdata, epochs = epochs, callbacks = callbacks, class_weight = weights)
    return model

