import sys
from time import sleep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from flask import Flask, render_template, request, Blueprint, url_for, send_file
import cv2
from tqdm import tqdm

sys.path.append('../src')
from src.ocr.normalization import word_normalization, letter_normalization
from src.ocr import page, words, characters
from src.ocr.helpers import implt, resize
from src.ocr.tfhelpers import Model
from src.ocr.datahelpers import idx2char

app = Flask(__name__)
# Global Variables
# IMG = r'data/pages/test4.jpg'  # 1, 2, 3
LANG = 'en'

MODEL_LOC_CHARS = r'models/char-clas/' + LANG + '/CharClassifier'
#MODEL_LOC_CTC = '../models/word-clas/CTC/Classifier2'


# Load Trained Model
CHARACTER_MODEL = Model(MODEL_LOC_CHARS)


# CTC_MODEL = Model(MODEL_LOC_CTC, 'word_prediction')


# Load image
# img = cv2.imread(IMG)
# image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# implt(image)
#
# Crop image and get bounding boxes
# crop = page.detection(image)
# implt(crop)
# boxes = words.detection(crop)
# lines = words.sort_words(boxes)


def recognise(img):
    """Recognition using character model"""
    # Pre-processing the word
    img = word_normalization(
        img,
        60,
        border=False,
        tilt=True,
        hyst_norm=True)

    # Separate letters
    img = cv2.copyMakeBorder(
        img,
        0, 0, 30, 30,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0])
    gaps = characters.segment(img, RNN=True)

    chars = []
    for i in range(len(gaps) - 1):
        char = img[:, gaps[i]:gaps[i + 1]]
        char, dim = letter_normalization(char, is_thresh=True, dim=True)
        # TODO Test different values
        if dim[0] > 4 and dim[1] > 4:
            chars.append(char.flatten())

    chars = np.array(chars)
    word = ''
    if len(chars) != 0:
        pred = CHARACTER_MODEL.run(chars)
        for c in pred:
            word += idx2char(c)

    return word


@app.route('/', methods=['POST', 'GET'])
def home_get():
    if request.method == 'POST':
        imagefile_2 = request.files['myfile']
        imagefile_2.save(f"static\images\content.jpeg")
        IMG = r'static\images\content.jpeg'
        # Load image
        img = cv2.imread(IMG)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # implt(image)

        # Crop image and get bounding boxes
        crop = page.detection(image)
        # implt(crop)
        boxes = words.detection(crop)
        lines = words.sort_words(boxes)
        with open('major1.txt', 'w') as f:
            for line in tqdm(lines):
                sleep(0.1)
                k = (" ".join([recognise(crop[y1:y2, x1:x2]) for (x1, y1, x2, y2) in line]))

                f.write(k)
                f.write('\n')
        f.close()

    return render_template('index.html')


@app.route('/download')
def download():
    path = 'major.txt'
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(debug=True)
