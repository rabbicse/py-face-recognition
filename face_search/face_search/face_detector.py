import glob
import os
import time
from multiprocessing.pool import ThreadPool

import cv2
import imutils
import numpy as np
from face_search.retina_face.retinaface import RetinaFace
from face_search.arc_face.face_recognition import FaceRecognition

from face_search.arc_face import preprocessor

FONT = cv2.FONT_HERSHEY_SIMPLEX


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def drawText(image, text):
    font_scale = 0.55
    margin = 5
    thickness = 2
    color = (255, 255, 255)

    size = cv2.getTextSize(text, FONT, font_scale, thickness)

    text_width = size[0][0]
    text_height = size[0][1]
    line_height = text_height + size[1] + margin

    x = image.shape[1] - margin - text_width
    y = margin + size[0][1] + line_height

    cv2.putText(image, text, (x, y), FONT, font_scale, color, thickness)


class FaceDetector:
    thresh = 0.8
    scales = [720, 1920]
    retina_face_model = os.path.abspath('models/retinaface-R50/R50')
    # face_detector = RetinaFace(retina_face_model, 0, 0, 'net3')
    arc_face_model = os.path.abspath('models/model-r100-arcface-ms1m-refine-v2/model-r100-ii/model-0000')
    face_detector = RetinaFace(retina_face_model, 0, 0, 'net3')
    recognizer = FaceRecognition(arc_face_model)

    def __init__(self):
        self.__embeddings = []
        self.recognizer.prepare(0)
        self.init_embeddings('train_data/')

    def __enter__(self):
        # self.face_detector = RetinaFace(self.retina_face_model, 0, 0, 'net3')
        # self.recognizer = FaceRecognition(self.arc_face_model)
        # self.recognizer.prepare(0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.recognizer
        del self.face_detector

    def init_embeddings(self, img_path):
        train_images = []
        for f in glob.glob('{}/*.jpg'.format(img_path)):
            train_images.append(f)

        for im_path in train_images:
            self.get_emb(im_path)

    def get_emb(self, im_path):
        # read image from file
        img = cv2.imread(im_path)

        # detect faces
        faces, landmarks = self.pre_process_img(img)

        # extract rect and landmark=5
        box = faces[0].astype(np.int)
        landmark5 = landmarks[0].astype(np.float32)

        # preprocess face before get embeddings
        crop_img = self.pre_process_face(img, box, landmark5)
        emb = self.recognizer.get_embedding(crop_img)

        v = im_path.split('/')[-1].replace('.jpg', '')
        self.__embeddings.append((emb.copy(), v))

        del crop_img
        del box
        del landmark5
        del emb
        del img

    def pre_process_img(self, img):
        try:
            # get image shape
            im_shape = img.shape

            target_size, max_size = self.scales
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            # if im_size_min>target_size or im_size_max>max_size:
            im_scale = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)

            print('im_scale', im_scale)
            faces, landmarks = self.face_detector.detect(img, self.thresh, scales=[im_scale], do_flip=False)

            return faces, landmarks
        except Exception as x:
            print(x)

    def pre_process_face(self, img, box, landmark5):
        try:

            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            ch = y2 - y1
            cw = x2 - x1
            margin = 0
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(x1 - margin / 2, 0)
            bb[1] = np.maximum(y1 - margin / 2, 0)
            bb[2] = np.minimum(x2 + margin / 2, img.shape[1])
            bb[3] = np.minimum(y2 + margin / 2, img.shape[0])
            ret = img[bb[1]:bb[3], bb[0]:bb[2], :]

            lan = [[l[0] - bb[0], l[1] - bb[1]] for l in landmark5]
            ln = np.array(lan, dtype=np.float32)

            crop_img = preprocessor.preprocess(ret, image_size=[112, 112], landmark=ln)
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            return crop_img
        except Exception as x:
            print(x)

    def process_image(self, image_path):
        # read image from file
        img = cv2.imread(image_path)

        # # get image shape
        im_shape = img.shape

        orig_img = img.copy()

        resized = imutils.resize(orig_img, height=720)
        r_shape = resized.shape

        faces, landmarks = self.pre_process_img(img)

        if faces.any():
            print('find', faces.shape[0], 'faces')
            for i in range(faces.shape[0]):
                box = faces[i].astype(np.int)

                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                landmark5 = landmarks[i].astype(np.float32)
                crop_img = self.pre_process_face(img, box, landmark5)

                emb = self.recognizer.get_embedding(crop_img)
                results = [(self.recognizer.compute_match(emb, c[0]), c[1]) for c in self.__embeddings]
                index = np.argmax([d[0] for d in results])
                dis = results[index]

                x01, x02, y01, y02 = x1 / im_shape[1], x2 / im_shape[1], y1 / im_shape[0], y2 / im_shape[0]

                rx01, rx02, ry01, ry02 = x01 * r_shape[1], x02 * r_shape[1], y01 * r_shape[0], y02 * r_shape[0]

                rx01, rx02, ry01, ry02 = int(rx01), int(rx02), int(ry01), int(ry02)

                text = '{}'.format(dis[1])
                size = cv2.getTextSize(text, FONT, 0.55, 2)
                color = (0, 0, 255)
                cv2.rectangle(resized, (int(rx01), int(ry01)), (int(rx02), int(ry02)), color, 2)
                if dis[0] < 0.35:
                    continue
                # Draw a label with a name below the face
                cv2.rectangle(resized, (rx01, ry02), (rx01 + size[0][0] + 6, ry02 + size[0][1] + 6), (0, 0, 255), cv2.FILLED)

                cv2.putText(resized, text, (rx01 + 6, ry02 + size[0][1]), FONT, 0.55, (255, 255, 255), 2)

        import base64
        img_str = base64.b64encode(cv2.imencode('.png', resized)[1])
        return img_str

    def process_images(self, image_path):
        # read image from file
        img = cv2.imread(image_path)

        # get image shape
        im_shape = img.shape

        target_size, max_size = self.scales
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # if im_size_min>target_size or im_size_max>max_size:
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        print('im_scale', im_scale)

        faces, landmarks = self.face_detector.detect(img, self.thresh, scales=[im_scale], do_flip=False)

        if faces.any():
            print('find', faces.shape[0], 'faces')
            for i in range(faces.shape[0]):
                # print('score', faces[i][4])
                box = faces[i].astype(np.int)
                # color = (255,0,0)
                color = (0, 0, 255)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                if landmarks is not None:
                    landmark5 = landmarks[i].astype(np.int)
                    # print(landmark.shape)
                    for l in range(landmark5.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

        cv2.imwrite('output_data/{}'.format(image_path.split('/')[-1]), img)
        # cv2.imshow('Images', img)

        # Hit 'q' on the keyboard to quit!
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def process_frames(self, img):
        # get image shape
        im_shape = img.shape

        target_size, max_size = self.scales
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # if im_size_min>target_size or im_size_max>max_size:
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        print('im_scale', im_scale)

        faces, landmarks = self.face_detector.detect(img, self.thresh, scales=[im_scale], do_flip=False)

        if faces.any():
            print('find', faces.shape[0], 'faces')
            for i in range(faces.shape[0]):
                # print('score', faces[i][4])
                box = faces[i].astype(np.int)
                # color = (255,0,0)
                color = (0, 0, 255)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                if landmarks is not None:
                    landmark5 = landmarks[i].astype(np.int)
                    # print(landmark.shape)
                    for l in range(landmark5.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

            cv2.imshow('Images', img)


@timeit
def recognize_images():
    with FaceDetector() as face_rec:
        test_images = []
        for f in glob.glob('test_data/*.jpg'):
            test_images.append(f)

        for im in test_images:
            face_rec.process_images(im)

        with ThreadPool(1) as pool:
            pool.map(face_rec.process_images, test_images)


def recognize_from_webcam():
    webcam = cv2.VideoCapture(0)
    with FaceDetector() as face_detector:
        while True:
            # Grab a single frame of video
            ret, frame = webcam.read()

            face_detector.process_frames(frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    recognize_images()
    # recognize_from_webcam()
