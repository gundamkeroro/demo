from flask import Flask, request, send_from_directory
from werkzeug.utils import secure_filename
import os

import Model
import ProcessImage

app = Flask(__name__, static_folder='gen_images')


@app.route('/')
def hello_world():
    return 'Hello World!'
#load model
# return a theano function
model = Model.load_model()

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/yuwangwen', methods=['POST'])
def yuwangwen():
    # TO DO check or corp image size?
    # the uploaded image should be 178 * 220 size image right now
    oridata = request.data
    if 'file' not in request.files:
        return 'Error!'
    # file is a object
    # for example
    # curl -X POST http://54.223.112.245:9527/ssdflow -H 'contentt/form-data' -F file=@/home/ruobo/Downloads/FireShot/flow_8.jpg
    # get a file
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join('/home/ubuntu/robbytest/dcgan-autoencoder/', filename))
        ProcessImage.prcessing() #creat hdf5 file
    else:
        print "not a valid form!"
        pass
    # Infer file using loaded model and hdf5 file
    path = "/home/ubuntu/robbytest/dcgan-autoencoder/faces_single_test.hdf5"
    result_path = Model.infer(path, model)
    # return result
    imgname = result_path.split('/')[-1]
    return send_from_directory(app.static_folder, imgname)

if __name__ == '__main__':
    app.run(host= '0.0.0.0')
