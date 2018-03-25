from __main__ import *
from utils.support import fr_utils
from utils.support import inception_blocks_v2
import glob
import signal

class Facenet(object):
    def _triplet_loss(self, y_true, y_pred, alpha = 0.2):
        """
        Implementation of the triplet loss as defined by formula (3)

        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor images, of shape (None, 128)
                positive -- the encodings for the positive images, of shape (None, 128)
                negative -- the encodings for the negative images, of shape (None, 128)

        Returns:
        loss -- real number, value of the loss
        """

        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

        pos_dist = tf.reduce_sum(np.square(tf.subtract(anchor, positive)), axis=-1)
        neg_dist = tf.reduce_sum(np.square(tf.subtract(anchor, negative)), axis=-1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist) , alpha)
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
        return loss

    def who_is_it(self, image_path, database, model):
        """
        Implements face recognition for the happy house by finding who is the person on the image_path image.

        Arguments:
        image_path -- path to an image
        database -- database containing image encodings along with the name of the person on the image
        model -- your Inception model instance in Keras

        Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
        """

        encoding = fr_utils.img_to_encoding(image_path, model)
        min_dist = 100

        for (name) in self.names:

            dist = np.linalg.norm(encoding - self.database[name])
            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > 0.7:
            print("Not in the database.")
        else:
            print ("it's " + str(identity) + ", the distance is " + str(min_dist))

        return min_dist, identity

    def verify(self, image_path, identity, database, model):
        """
        Function that verifies if the person on the "image_path" image is "identity".

        Arguments:
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras

        Returns:
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
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
        for image in glob.glob("/images/*"):
            self.database[image] = self.img_to_encoding(image, self.FRmodel)
            self.names.append(image)
    def get_frame(self, name, cascade):
        vc = cv2.VideoCapture(0)
        self.vc = vc
        if vc.isOpened():
            is_capturing, _ = vc.read()
        else:
            is_capturing = False

        imgs = []
        signal.signal(signal.SIGINT, self._signal_handler)
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
#                 print(left, right)
#                 print(top, bottom)
#                 cv2.rectangle(frame,
#                               (left-1, bottom-1),
#                               (right+1, top+1),
#                               (255, 0, 0), thickness=2)
#                 print (left, top)
#                 print(right, bottom)
                frame = frame[bottom-1:top+1, left-1:right+1]

            resized_image = cv2.resize(frame, (96, 96))
            cv2.imwrite('image.png', resized_image)
            vc.release()
            print(Facenet.who_is_it('image.png', self.database, self.FRmodel))
            #display.clear_output(wait=True)


    def __init__(self):
        self.FRmodel = inception_blocks_v2.faceRecoModel(input_shape=(3, 96, 96))
        print("Total Params:", self.FRmodel.count_params())

        self.FRmodel.compile(optimizer = 'adam', loss = Facenet._triplet_loss, metrics = ['accuracy'])
        fr_utils.load_weights_from_FaceNet(self.FRmodel)

        self.img_to_encoding = fr_utils.img_to_encoding
        self.database = {}
        self.names = []
        loadDb()
        # self.database["danielle"] = self.img_to_encoding("images/danielle.png", self.FRmodel)
        # self.database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
        # self.database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
        # self.database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
        # self.database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
        # self.database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
        # self.database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
        # self.database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
        # self.database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
        # self.database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
        # self.database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
        # self.database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)
