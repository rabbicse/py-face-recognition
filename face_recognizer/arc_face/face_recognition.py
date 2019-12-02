from __future__ import division
import mxnet as mx
import numpy as np
from skimage import transform as trans
import sklearn
from sklearn import preprocessing
from numpy.linalg import norm
import cv2


class FaceRecognition:
    def __init__(self, param_file):
        self.param_file = param_file
        self.image_size = (112, 112)
        self.src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

    def prepare(self, ctx_id):
        if self.param_file:
            pos = self.param_file.rfind('-')
            prefix = self.param_file[0:pos]
            pos2 = self.param_file.rfind('.')

            epoch = int(self.param_file[pos + 1:pos2])
            sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
            all_layers = sym.get_internals()
            sym = all_layers['fc1_output']
            if ctx_id >= 0:
                ctx = mx.gpu(ctx_id)
            else:
                ctx = mx.cpu()

            model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
            data_shape = (1, 3) + self.image_size
            model.bind(data_shapes=[('data', data_shape)])
            model.set_params(arg_params, aux_params)
            # #warmup
            # data = mx.nd.zeros(shape=data_shape)
            # db = mx.io.DataBatch(data=(data,))
            # model.forward(db, is_train=False)
            # embedding = model.get_outputs()[0].asnumpy()
            self.model = model

    def get_embedding(self, img, rgb_convert=False):
        assert self.param_file and self.model
        assert img.shape[2] == 3 and img.shape[0:2] == self.image_size

        if rgb_convert:
            data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            data = img

        data = np.transpose(data, (2, 0, 1))
        data = np.expand_dims(data, axis=0)
        data = mx.nd.array(data)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding

        # Normalise embedding obtained from forward pass to unit vector
        # embedding = self.model.get_outputs()[0].squeeze()
        # embedding /= embedding.norm()
        # return embedding

    def get(self, rimg, landmark):
        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg, M, (self.image_size[1], self.image_size[0]), borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        input_blob = np.zeros((1, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding

    def compute_sim(self, img1, img2):
        emb1 = self.get_embedding(img1).flatten()
        emb2 = self.get_embedding(img2).flatten()
        sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        return sim

    def compute_match(self, emb1, emb2):
        # emb1 = emb1.flatten()
        # emb2 = emb2.flatten()
        sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        return sim

        # sim = np.dot(emb1, emb2.T)
        # return sim
