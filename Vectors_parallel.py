import base64
import io
import logging

import cv2
from atpbar import atpbar
import tensorflow as tf
import p_tqdm
import keras
import numpy as np
import progressbar
from PIL import Image
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from pymongo import MongoClient

#String constants
str_image = 'Image'
str_id = 'DocId'
str_vector = 'ImgVector'

# Take in base64 string and return a 3D RGB image (numpy array)
def stringToRGB(base64_string):
    im = Image.open(io.BytesIO(base64.b64decode(base64_string)))
    im = np.array(im)
    if len(im.shape) < 3:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    if len(im.shape) != 3:
        logging.error("Please use RGB (3 channels) or grayscale (1 channel) images.")
    return im


# Take in a numpy array and return a base64 string
def arraytostring(array):
    return base64.b64encode(array)


# Take in a base64 string and return a numpy array
def stringtoarray(string):
    return np.frombuffer(base64.decodebytes(string), dtype=np.float32)


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


def img_to_vec(model, init_image):
    img = reshape_resize(init_image)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model.predict(img).ravel()

def chunks(l, n):
    for i in range(n):
        yield l[i::n]


class Vectorize:

    def __init__(self, modelname):
        # self.model = keras.models.load_model(modelname, compile=False)
        logging.info('Neural network has been initialized')
        self.vectors = []
        self.client = MongoClient('localhost', 27017)
        logging.info("Successful connection to the database.")
        self.db = self.client['Patents']
        self.vectors = self.db['Vectors']
        print('Vectorize class initialized.')

    # Outdated function
    def all_files_to_vec(self):
        logging.info("Rewriting vectors database.")
        self.delete_vecs()
        logging.info("Rewriting vectors database: database emptied.")
        max_value = self.db.tm.count_documents({})
        bar = progressbar.ProgressBar(max_value=max_value)
        i = 0
        logging.info("Rewriting vectors database: iteration through database began.")
        for doc in self.db.tm.find({}, batch_size=200):
            docid = doc[str_id]
            vector = img_to_vec(stringToRGB(doc[str_image]))
            to_insert = {str_id: docid, str_vector: arraytostring(vector)}
            self.vectors.insert_one(to_insert)
            bar.update(i)
            i += 1
            if i >= max_value:
                break
        logging.info("Rewriting vectors database: iteration through database done. All vectors updated.")

    def delete_vecs(self):
        self.vectors.delete_many({})


def update_vecs(chunk, model):
    client = MongoClient('localhost', 27017)
    db = client['Patents']
    vectors = db['Vectors']

    logging.info("Updating vectors database.")
    inserted = 0
    logging.info("Updating vectors database: iteration through database began.")
    for i in atpbar(range(len(chunk))):
        doc = db.tm.find_one(chunk[i])
        docid = doc[str_id]
        results = vectors.find_one({str_id: docid})
        if results is None:
            vector = img_to_vec(model, stringToRGB(doc[str_image]))
            to_insert = {str_id: docid, str_vector: arraytostring(vector)}
            vectors.insert_one(to_insert)
            inserted += 1
    # print('# of processed vectors: ', vectors.count())
    logging.info("Updating vectors database: iteration through database done. {} documents inserted".format(inserted))


if __name__ == "__main__":
    NUM_THREADS = 4

    client = MongoClient('localhost', 27017)
    db = client['Patents']
    vectors = db['Vectors']
    ids = db.tm.find().distinct('_id')
    chunks_ids = chunks(ids, NUM_THREADS)
    print('Total # of documents: ', db.tm.count_documents({}))
    print('Inintial # of vectors: ', vectors.tm.count_documents({}))

    # vec = Vectorize(modelname='model.h5')#, imgdir='database_logos_part')
    # vec.delete_vecs()
    # exit()

    models = []
    for i in range(NUM_THREADS):
        models.append(keras.models.load_model('model.h5', compile=False))

    # update_partial = partial(update_vecs, model=model_nn)
    # res = process_map(update_partial, chunks_ids)
    res = p_tqdm.p_map(update_vecs, list(chunks_ids), models)

    # i = 0
    # threads = list()
    # for model, chunk in zip(models, chunks_ids):
    #     print("Main    : create and start thread %d.", i)
    #     x = threading.Thread(target=update_vecs, args=(chunk, model), daemon=True)
    #     threads.append(x)
    #     x.start()
    #     i += 1

    # for index, thread in enumerate(threads):
    #     print("Main    : before joining thread %d.", index)
    #     thread.join()
    #     print("Main    : thread %d done", index)