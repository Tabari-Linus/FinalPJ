import tensorflow as tf
import cv2
import numpy as np

# 1. Load the TensorFlow model
model_path = '/home/ltabari/Desktop/FInal year/project trials/Final Project/fruit_model.tflite'
model = tf.keras.models.load_model(model_path)


# Creating parameters for loader
batch_size = 32
img_height = 180
img_width = 180

# 2. Preprocess the input image
test_image = "/home/ltabari/Desktop/FInal year/project trials/one/dataset/fruit_test_data/banana/rot.webp"
input_shape = (img_height, img_width)  # Replace with the desired input size of the model

# Read and preprocess the image
input_image = cv2.imread(test_image)
input_image = cv2.resize(input_image, input_shape)
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension if needed

# 3. Run inference
predictions = model.predict(input_image)

# Optionally, you can post-process the predictions based on your model's output format or task.
# For example, for classification tasks, you can use argmax to get the predicted class.

# Example for classification task:
predicted_class_index = np.argmax(predictions[0])
predicted_class_name = 'class_labels'  # Replace with your class labels
predicted_class = predicted_class_name[predicted_class_index]

print("Predicted Class:", predicted_class)
