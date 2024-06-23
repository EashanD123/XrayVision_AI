from flask import Flask, redirect, url_for, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('home.html')

@app.route('/possible_diseases')
def possible_diseases():
    return render_template('possible_diseases.html')

@app.route('/test_disease')
def test_disease():
    return render_template('test_disease.html')

@app.route('/test_disease1')
def test_disease1():
    return render_template('test_disease1.html')

@app.route('/test_disease2')
def test_disease2():
    return render_template('test_disease2.html')

@app.route('/test_disease3')
def test_disease3():
    return render_template('test_disease3.html')

def base64_to_image(base64_string):
    # Remove the data URI prefix if present
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]

    # Decode the Base64 string into bytes
    image_bytes = base64.b64decode(base64_string)
    return image_bytes


def create_image_from_bytes(image_bytes):
    # Create a BytesIO object to handle the image data
    image_stream = BytesIO(image_bytes)

    # Open the image using Pillow (PIL)
    image = Image.open(image_stream)
    return image


@app.route('/predict', methods=['POST'])
def predict():
    test_dir = 'content/chest-xray-pneumonia/chest_xray/test'
    m = "pneumonia_model3.keras"
    classes = ['NORMAL', 'PNEUMONIA']

    data = request.get_json(force=True)
    disease = np.array(data['disease']).reshape(1, -1)
    test_image = np.array(data['test_image']).reshape(1, -1)
    if disease != "pneumonia":
        test_dir = 'content/turbuculosis_dataset'
        m = "turbuculosis_model2.keras"
        classes = ['NORMAL', 'TURBUCULOSIS']

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')

    image_bytes = base64_to_image(test_image)

    # Create an image from bytes
    img = create_image_from_bytes(image_bytes)
    img.save("img.jpeg")
    img = tf.keras.utils.load_img("img.jpeg", target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    load_model = tf.keras.models.load_model(m)
    result = load_model.predict(img_array)
    class_index = tf.argmax(result, axis=1).numpy()[0]

    accuracy = load_model.evaluate(test_generator)

    print(classes[class_index])

    response = {'prediction': {'result': classes[class_index], 'accuracy': accuracy}}
    print(response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
