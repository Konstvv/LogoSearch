import os
import numpy as np
import cv2
from Vectors import Vectorize, stringToRGB
from pymongo import MongoClient
import keras
import matplotlib.pyplot as plt
from keras.preprocessing import image
from sklearn.metrics import roc_curve, roc_auc_score


def delete_text_all(modelname):
    model_text = keras.models.load_model(modelname, compile=False)
    client = MongoClient('localhost', 27017)
    db = client['Patents']
    vectors = db['Vectors']
    str_image = 'Image'
    str_id = 'DocId'
    str_vector = 'ImgVector'

    for doc in db.tm.find({}):
        arr = []
        img = stringToRGB(doc[str_image])
        img = Vectorize.reshape_resize(img)
        arr.append(img)
        arr = np.asarray(arr).astype('float32')
        pred = model_text.predict(arr)
        if pred[0] > 0.95:
            # db.tm.remove(doc)
            plt.figure()
            plt.imshow(img)
            plt.show()

modelname = 'save_at_16.h5'

delete_text_all(modelname)
# model_text = keras.models.load_model(modelname, compile=False)
#
# data = []
# labels = []
#
# for subdir, dirs, files in os.walk('LogoLogos'):
#     for file in files:
#         img = np.array(cv2.imread((os.path.join(subdir, file))))
#         print(type(img))
#         img = Vectorize.reshape_resize(img)
#         data.append(img)
#         labels.append(0)
#
# for subdir, dirs, files in os.walk('TextLogos'):
#     for file in files:
#         img = np.array(cv2.imread((os.path.join(subdir, file))))
#         img = Vectorize.reshape_resize(img)
#         data.append(img)
#         labels.append(1)
#
# data = np.asarray(data).astype('float32')
# labels = np.asarray(labels).astype('float32')
#
# print('prediction began')
# pred = model_text.predict(data)
# # plt.plot(pred)
#
# fpr, tpr, t = roc_curve(labels, pred)
# score = roc_auc_score(labels, pred)
# plt.plot(fpr, tpr, label='score: {}'.format(score))
# plt.legend()
# plt.show()
