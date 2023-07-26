import numpy as np
import tensorflow as tf
import cv2

saved_model_dir = '/home/ltabari/Desktop/FInal year/project trials/Final Project/fruit_model.tflite'
net = cv2.dnn.readNetFromTensorflow(saved_model_dir)

img_height = 180
img_width = 180
test_image = "/home/ltabari/Desktop/FInal year/project trials/one/dataset/fruit_test_data/tomatoes/1.jpeg"


input_shape = (img_height, img_height)  # Replace with the desired input size of the model
input_image = cv2.resize(test_image, input_shape)
input_data = input_image.astype('float32') / 255.0  # Normalize if needed


# Pass the input data to the model
net.setInput(input_data)

# Run forward pass to get predictions
output_data = net.forward()

print(output_data)