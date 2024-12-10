#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from io import BytesIO
from urllib import request
import tflite_runtime.interpreter as tflite
from PIL import Image

# Load Model
MODEL_NAME = 'model_2024_hairstyle_v2.tflite'
interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Clases
classes = [
    'curly',
    'straight'
]

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def prepare_input(x):
    return x / 255.0

def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(200, 200))

    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()
    print('float_predictions', float_predictions)
    
    return dict(zip(classes, float_predictions))

def lambda_handler(event, context):
    url = event["url"]
    print(url)
    pred = predict(url)
    result = {"prediction": pred}
    return result