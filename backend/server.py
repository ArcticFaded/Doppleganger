from utils import *
from utils import facenet
from flask import Flask, send_file, request
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

comparer = facenet.Facenet()

@app.route("/v1/capture")
def capture():
    comparer.get_frame("mine",  cv2.CascadeClassifier('utils/support/haarcascade_frontalface_alt2.xml'))

@app.route("/v1/compare", methods=['GET','POST'])
def compare():
    if request.method == 'POST':
        imagefile = request.files['image']
        f = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
        imagefile.save(f)
        return comparer.who_is_it(f, comparer.database, comparer.FRmodel)
        #return send_file(f, mimetype='image/gif')
    else:
        return send_file('images\\andrew.jpg', mimetype='image/gif')

if __name__ == '__main__':
    app.run(debug=True)
