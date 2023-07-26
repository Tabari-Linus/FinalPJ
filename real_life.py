import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='/home/ltabari/Desktop/FInal year/project trials/Final Project/fruit_model.tflite')
interpreter.allocate_tensors()

# Get the model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess input data (assuming image_path is the path to your input image)
input_image = cv2.imread(image_path)
input_image = cv2.resize(input_image, (input_shape_width, input_shape_height))
input_data = np.expand_dims(input_image, axis=0)
input_data = (input_data - mean) / std  # Normalize if needed

# Set the input tensor to the interpreter
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor from the interpreter
output_data = interpreter.get_tensor(output_details[0]['index'])

# Post-process the output data as needed
# ...

# Use OpenCV for displaying or additional processing if desired
# ...
