from __future__ import division
import mxnet as mx
import numpy as np
from numpy.linalg import norm
import cv2


class FaceRecognition:
    def __init__(self, param_file):
        self.param_file = param_file
        self.image_size = (112, 112)

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
        return embedding

    def compute_sim(self, img1, img2):
        emb1 = self.get_embedding(img1).flatten()
        emb2 = self.get_embedding(img2).flatten()
        sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        return sim

    def compute_match(self, emb1, emb2):
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        return sim
