#!/usr/bin/env python
# coding: utf-8
#<img src="{{ user_image }}" alt="User Image">


from imageai.Detection import ObjectDetection
from PIL import Image
import glob
import os
import flask
from flask import Flask, redirect, url_for, request, render_template
from flask import jsonify
from flask_caching import Cache
from flask_uploads import UploadSet, configure_uploads, ALL
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import base64
import random


PEOPLE_FOLDER = os.path.join('static', 'people_photo')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
#cache.init_app(app)		
files = UploadSet('files', ALL)
app.config['UPLOADED_FILES_DEST'] = './static/uploaded_imgs'
configure_uploads(app, files)
#OUTPUT_PATH = './static/result_imgs/detected'
roots_to_clear = [app.config['UPLOADED_FILES_DEST'], './static/result_imgs']

#PEOPLE_FOLDER = os.path.join('static', 'people_photo')

#app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("resnet50_coco_best_v2.0.1.h5")
detector.loadModel()
custom_objects = detector.CustomObjects(person=True, car=True)



@app.route('/', methods=['GET', 'POST'])
def index():
    for root_to_clear in roots_to_clear:
        for root, dirs, files in os.walk(root_to_clear):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))
    return render_template('home.html')


@app.route('/preview', methods=['GET', 'POST'])
def upload_and_preview():
    if request.method == 'POST' and 'media' in request.files:
        _ = files.save(request.files['media'])
        list_of_files = glob.glob(os.path.join(app.config['UPLOADED_FILES_DEST'], '*'))
        latest_file = max(list_of_files, key=os.path.getctime)
        global image_to_process
        image_to_process = latest_file
        print(image_to_process)
        print(latest_file)
        img_to_render = os.path.join('..', latest_file)
        img_to_render = img_to_render.replace('\\', '/')
    else:
        img_to_render = '../static/images/no_image.png'
    paths = ['../static/images/no_image.png']

    return render_template('preview.html', img_name=img_to_render, img_paths=paths)


@app.route('/')
@app.route('/index',methods=['GET', 'POST'])
def obj_detect():
	apple=random.randint(0,100000000)
	naming='static/people_photo/' + str(apple) + '.png'
	naming_indx=str(apple) + '.png'
	print(naming)
	print(naming_indx)
	detections = detector.detectCustomObjectsFromImage(input_image=image_to_process,output_image_path=naming,custom_objects=custom_objects, minimum_percentage_probability=65,thread_safe=True)
	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], naming_indx)
	return render_template("index.html", user_image = full_filename)

if __name__ == '__main__':
    app.run(debug=True)