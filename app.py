import numpy as np
from flask import Flask, request, jsonify, render_template
from flask import Blueprint, render_template,redirect,url_for,request
import sys, json
import random
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#import argparse
from pickle import dump, load
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('index.html')

@app.route('/process', methods = ['POST'])
def read_in():
    id=request.form.get("ID")
    if request.method == 'POST':
        file=request.files['file']
        if not os.path.isdir('static'):
            os.mkdir('static')
        filepath=os.path.join('static',"img.jpg")
        file.save(filepath)
    #Since our input would only be having one line, parse our JSON data from that
    return redirect(url_for('caption'))

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
         if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

@app.route('/caption')
def caption():
    img_path = "img.jpg"
    max_length = 32
    tokenizer = load(open("tokenizer.pkl","rb"))
    model = load_model('models/model_9.h5')
    xception_model = Xception(include_top=False, pooling="avg")
    photo = extract_features(img_path, xception_model)
    #img = Image.open(img_path)
    description = generate_desc(model, tokenizer, photo, max_length)
    return render_template('index.html', text = description[6:-4])

if __name__ == "__main__":
    app.run(debug=True)