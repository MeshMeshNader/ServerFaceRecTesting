import face_recognition
import numpy as np
import urllib.request
import json
import io
import base64
from flask import Flask, request, abort
import logging

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
img = ''

TOLERANCE = 0.59
MODEL = 'cnn'  # 'hog' or 'cnn' - CUDA accelerated (if available) deep-learning pretrained model


@app.route("/FaceRecognitionTestingGetImage", methods=['POST'])
def Get_Image():
    if not request.json or 'image' not in request.json:
        abort(400)

    im_b64 = request.json['image']
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))
    global img
    img = io.BytesIO(img_bytes)

    result_dict = {"output": 'Done saving image'}
    return result_dict


@app.route("/FaceRecognitionTesting", methods=['POST'])
def Recognize_Face():
    global img
    if not request.json or 'encodings' not in request.json:
        abort(400)

    if not request.json or 'url' not in request.json:
        abort(400)

    all_face_encodings = json.loads(request.json['encodings'])

    img_url = request.json['url']

    known_names = list(all_face_encodings.keys())
    known_faces = np.array(list(all_face_encodings.values()))

    # response = urllib.request.urlopen(img_url)
    # image = face_recognition.load_image_file(response)

    image = face_recognition.load_image_file(img)

    locations = face_recognition.face_locations(image, model=MODEL)

    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):

        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None

        if True in results:
            match = known_names[results.index(True)]
            result = "I Found " + match

        check = all(element == False for element in results)

        if check:
            result = "I Found Unknown Person"

    result_dict = {"output": result}
    return result_dict


# app.run(debug=True)
