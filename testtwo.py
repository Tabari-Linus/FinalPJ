import tensorflow as tf
import pathlib
import numpy as np

dataset_url = "/home/ltabari/Desktop/FInal year/project trials/Final Project/dataset/fruit_data/"
data_dir = pathlib.Path(dataset_url)



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

class_names = train_ds.class_names

# Creating parameters for loader
batch_size = 32
img_height = 180
img_width = 180

test_image = "/home/ltabari/Desktop/FInal year/project trials/one/dataset/fruit_test_data/banana/rot.webp"
t_image = test_image

img = tf.keras.utils.load_img(
    t_image, target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

loaded_model = tf.keras.models.load_model('/home/ltabari/Desktop/FInal year/project trials/Final Project//model.h5')
predictions = loaded_model.predict(img_array)


score = tf.nn.softmax(predictions[0])

print(
    "This is  {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
)