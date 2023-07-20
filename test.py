import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# classes in the dataset
""" fruit_data/
  banana/
  orange/
  tomatoes/
  """


# importing the data into the local

dataset_url = "/home/ltabari/Desktop/FInal year/project trials/Final Project/dataset/fruit_data/"
data_dir = pathlib.Path(dataset_url)


banana = list(data_dir.glob('Rotten_banana/*'))
print(banana)

# Creating parameters for loader
batch_size = 32
img_height = 180
img_width = 180

# using a validate split of 80%
# training set
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# validating set
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# getting the data class names attributes 
class_names = train_ds.class_names

print(class_names)