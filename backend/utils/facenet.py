from __main__ import *
from utils.support import fr_utils
from utils.support import inception_blocks_v2
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
#from inception_blocks_v2 import *
import signal
from IPython import display
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize


np.set_printoptions(threshold=np.nan)
import glob

class Facenet(object):
    def triplet_loss(self, y_true, y_pred, alpha = 0.2):
        """
        Implementation of the triplet loss.
        """

        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

        pos_dist = tf.reduce_sum(np.square(tf.subtract(anchor, positive)), axis=-1)
        neg_dist = tf.reduce_sum(np.square(tf.subtract(anchor, negative)), axis=-1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist) , alpha)
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
        return loss

    def who_is_it(self, image_path, database, model):
        """
        performs verify against a database
        """

        encoding = fr_utils.img_to_encoding(image_path, model)
        min_dist = 100
        identity = "default"
        for (name) in self.names:

            dist = np.linalg.norm(encoding - database[name][0])
            if dist < min_dist:
                min_dist = dist
                identity = name

        # if min_dist > 0.7:
        #     print("Not in the database.")
        # else:
        #     print ("it's " + str(identity) + ", the distance is " + str(min_dist))

        return min_dist, identity, database[identity][1]

    def verify(self, image_path, identity, database, model):
        """
        Function that verifies if the person on the "image_path" image is "identity".
        """

        encoding = img_to_encoding(image_path, model)
        dist = np.linalg.norm(encoding - database[identity])

        if dist < 0.7:
            print("It's " + str(identity) + ", welcome home!")
            door_open = True
        else:
            print("It's not " + str(identity) + ", please go away")
            door_open = False

        return dist, door_open
    def loadDb(self):
        foodchoice = ['Slow-Cooked Salted Pork',
        'Steamed Honey & Nuts Lamb',
        'Gentle-Fried Curry of Frog',
        'Gentle-Fried Raspberry & Peanut Trout',
        'Deep-Fried Garlic & Rosemary Flatbread',
        'Cooked Cinnamon Tofu',
        'Coffee and Cashew Fudge',
        'Red Wine and Mandarin Fudge',
        'Peanut Yogurt',
        'Saffron Candy']
        for image in glob.glob(os.getcwd() + "\\images\\*"):
            self.database[image] = [self.img_to_encoding('images\\' + image.split('\\')[-1], self.FRmodel), np.random.choice(foodchoice, 1)]
            self.names.append(image)

    def get_frame(self, name, cascade):
        cascade = cv2.CascadeClassifier(cascade)
        vc = cv2.VideoCapture(0)
        self.vc = vc
        if vc.isOpened():
            is_capturing, _ = vc.read()
        else:
            is_capturing = False

        imgs = []
        while is_capturing:
            is_capturing, frame = vc.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = cascade.detectMultiScale(frame,
                                         scaleFactor=1.1,
                                         minNeighbors=3,
                                         minSize=(100, 100))
            if len(faces) != 0:
                face = faces[0]
                (x, y, w, h) = face
                left = x - self.margin // 2
                right = x + w + self.margin // 2
                bottom = y - self.margin // 2
                top = y + h + self.margin // 2
                img = resize(frame[bottom:top, left:right, :],
                             (160, 160), mode='reflect')
                imgs.append(img)
                frame = frame[bottom-1:top+1, left-1:right+1]

            resized_image = cv2.resize(frame, (96, 96))
            cv2.imwrite('image.png', resized_image)
            vc.release()
            is_capturing, frame = vc.read()
            return (self.who_is_it('image.png', self.database, self.FRmodel))
            #display.clear_output(wait=True)


    def __init__(self):
        self.FRmodel = inception_blocks_v2.faceRecoModel(input_shape=(3, 96, 96))
        print("Total Params:", self.FRmodel.count_params())

        self.FRmodel.compile(optimizer = 'adam', loss = self.triplet_loss, metrics = ['accuracy'])
        fr_utils.load_weights_from_FaceNet(self.FRmodel)

        self.img_to_encoding = fr_utils.img_to_encoding
        self.database = {}
        self.names = []
        self.margin = 10
        self.loadDb()
