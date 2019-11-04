"""1. Getting Started with Pre-trained Model from RetinaFace
=======================================================


In this tutorial, we will demonstrate how to load a pre-trained model from :ref:`insightface-model-zoo`
and detect faces from images.

Step by Step
------------------

Let's first try out a pre-trained retinaface model with a few lines of python code.

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
    return image
################################################################
#
# Then, we download and show the example image:

url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
img = url_to_image(url)



################################################################
# Get RetinaFace model by its name.
#

model = insightface.model_zoo.get_model('retinaface_r50_v1')

################################################################
# Prepare the enviorment, to use CPU to detect the faces for all incoming images.(Change ctxid to a positive integer if you have GPUs)
# The nms threshold is set to 0.4 in this example.
#
model.prepare(ctx_id = -1, nms=0.4)

################################################################
# do face detection on input image, with original resolution and threshold 0.5.
#
bbox, landmark = model.detect(img, threshold=0.5, scale=1.0)

