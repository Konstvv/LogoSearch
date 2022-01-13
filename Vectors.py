from keras.applications import VGG19
from keras.engine import Model
import cv2
import keras
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from PIL import Image
import base64
import io
import numpy as np
import pickle
import progressbar
from pymongo import MongoClient
import argparse
import logging
import matplotlib.pyplot as plt

# Take in base64 string and return a 3D RGB image (numpy array)
def stringToRGB(base64_string):
    im = Image.open(io.BytesIO(base64.b64decode(base64_string)))
    im = np.array(im)
    if len(im.shape) == 3:
        if im.shape[2] == 3:
            pass
        elif im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        else:
            logging.error("Please use RGB (3 channels), RGBA (4 channels), or grayscale (1 channel) images.")
    elif len(im.shape) < 3:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    else:
        logging.error("Please use RGB (3 channels), RGBA (4 channels), or grayscale (1 channel) images.")
    return im


# Take in a numpy array and return a base64 string
def arraytostring(array):
    return base64.b64encode(array)

# Take in a base64 string and return a numpy array
def stringtoarray(string):
    return np.frombuffer(base64.decodebytes(string), dtype=np.float32)

class Vectorize:

    def __init__(self, modelname):
        self.model = keras.models.load_model(modelname, compile=False)
        logging.info('Neural network has been initialized')
        self.vectors = []
        try:
            self.client = MongoClient('localhost', 27017)
            self.db = self.client['Patents']
            self.vectors = self.db['Vectors']
            self.str_image = 'Image'
            self.str_id = 'DocId'
            self.str_vector = 'ImgVector'
        except Exception as exc:
            logging.error("An error occurred: {}".format(exc))

    @staticmethod
    def reshape_resize(img):
        if img is None:
            logging.error('No image was presented to the preprocessing function.')
        if img.shape[0] != img.shape[1]:
            if img.shape[0] > img.shape[1]:
                bordersize = int((img.shape[0] - img.shape[1]) / 2)
                border = cv2.copyMakeBorder(
                    img,
                    top=0,
                    bottom=0,
                    left=bordersize,
                    right=bordersize,
                    borderType=cv2.BORDER_REPLICATE
                )
            else:
                bordersize = int((img.shape[1] - img.shape[0]) / 2)
                border = cv2.copyMakeBorder(
                    img,
                    top=bordersize,
                    bottom=bordersize,
                    left=0,
                    right=0,
                    borderType=cv2.BORDER_REPLICATE
                )
        else:
            border = img
        resized = cv2.resize(border, (224, 224), interpolation=cv2.INTER_LINEAR)
        return resized

    def img_to_vec(self, init_image):
        img = self.reshape_resize(init_image)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return self.model.predict(img).ravel()

    def delete_vecs(self):
        self.vectors.delete_many({})

    def update_vecs_v2_c_sharp(self):
        logging.info("Updating vectors database.")
        ids_vectors = self.vectors.find().distinct('DocId')
        max_value = self.db.tm.count_documents({"DocId": {"$nin": ids_vectors}})
        bar = progressbar.ProgressBar(max_value=max_value)
        i = 0
        inserted = 0
        logging.info("Updating vectors database: iteration through database began.")
        for doc in self.db.tm.find({"DocId": {"$nin": ids_vectors}}, batch_size=1000):
            docid = doc[self.str_id]
            vector = self.img_to_vec(stringToRGB(doc[self.str_image]))
            vector = vector.tolist()
            to_insert = {self.str_id: docid, self.str_vector: vector}
            self.vectors.insert_one(to_insert)
            inserted += 1
            bar.update(i)
            i += 1
            if i >= max_value:
                break
        logging.info("Updating vectors database: iteration through database done. {} documents inserted".format(inserted))

    def save_vectors(self, name='vectors.pickle'):
        with open(name, 'wb') as f:
            pickle.dump(self.vectors, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', default=False, action='store_true', help='to overwrite all existing vectors')
    parser.add_argument('--update', dest='all', action='store_false')
    args = parser.parse_args()

    vec = Vectorize(modelname='model.h5')

    if args.all:
        vec.delete_vecs()
    vec.update_vecs_v2_c_sharp()