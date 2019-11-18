"""1. Getting Started with Pre-trained Model from ArcFace
=======================================================


In this tutorial, we will demonstrate how to load a pre-trained model from :ref:`insightface-model-zoo`
and get face embedding from images.

Step by Step
------------------

Let's first try out a pre-trained arcface model with a few lines of python code.

First, please follow the `installation guide <../../index.html#installation>`__
to install ``MXNet`` and ``insightface`` if you haven't done so yet.
"""

import insightface
import urllib
import urllib.request
import cv2
import numpy as np


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image


################################################################
#
# Then, we download and show the example image:

url = 'https://github.com/deepinsight/insightface/raw/master/deploy/Tom_Hanks_54745.png'
img = url_to_image(url)

################################################################
# Get Arcface model by its name.
#

model = insightface.model_zoo.get_model('arcface_r100_v1')

################################################################
# Prepare the enviorment, to use CPU to recognize the incoming face image.(Change ctxid to a positive integer if you have GPUs)
#
model.prepare(ctx_id=0)

################################################################
# Do face recognition, get the embedding vector.
#
emb = model.get_embedding(img)


