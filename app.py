from flask import Flask, render_template, request, redirect, url_for, send_file, flash, redirect, session
import os
import zipfile
from werkzeug.utils import secure_filename
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import shutil
import logging
from statistics import mode
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import openai

tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

UPD_FLD = 'static/uploads/'

app = Flask(__name__)

UPLOAD_FOLDER = os.path.dirname(os.path.realpath(__file__))
ALLOWED_EXTENSIONS = set(['zip','txt', 'pdf', 'png', 'jpg', 'jpeg'])
str = "click to upload a file"

app.config['UPLOAD_FOLDER'] = UPD_FLD
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'This is your secret key to utilize session in Flask'

classes = { 1 : 'Melanocytic nevi',
    		2 : 'Melanoma',
    		3 : 'Benign keratosis-like lesions ',
    		4 : 'Basal cell carcinoma',
    		5 : 'Actinic keratoses',
    		6 : 'Vascular lesions',
    		7 : 'Dermatofibroma'
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/main_page')
def main_page():
    return render_template("main_page.html")

def model():
	class_count = 7
	img_shape = (310, 640, 3)
	lr=.001
	base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
	base_model.trainable= True
	x = base_model.output
	x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
	x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
					bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
	x = Dropout(rate=.4, seed=123)(x)       
	output = Dense(class_count, activation='softmax')(x)
	model = Model(inputs=base_model.input, outputs=output)
	model.compile(Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy']) 
	model.load_weights("static/models/dermi-pre.h5")
	return model


@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']

	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

		skin_model  = model()
		img = cv2.imread(file_path)
		img = cv2.resize(img,(640,310))
		img = np.expand_dims(img,0)
		res = classes[np.argmax(skin_model.predict(img))]
		print(res)
		flash(res,"disease")


		select_lang = request.form.get('lang')
		print(select_lang)
		openai.api_key = 'enter openai api key'
		input_topic = f"{res} skin diseases"
		t = openai.Completion.create(
		engine='text-davinci-003',
		max_tokens=2048,
		prompt= f'Explain about {input_topic} in {select_lang} : Give the symptoms of the disease ,cure for the disease and home made first aid of the disease and tell me about the disease')
		t = (t['choices'][0]['text']).split('\n')[2:]
		response = [text for text in t if text != ""]
		for res in response:
			flash(res,"result")
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)
	
@app.route('/')
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
	# HEX=101