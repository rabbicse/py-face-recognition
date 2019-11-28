import glob
import os
import time
from multiprocessing.pool import ThreadPool

import cv2
import imutils
import numpy as np
from retina_face.retinaface import RetinaFace
from arc_face.face_recognition import FaceRecognition

# count = 1

# detector = RetinaFace(retina_face_model, 0, 0, 'net3')

# test_img = './test_data/weeding.jpg'


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
    scales = [1280, 1920]
    retina_face_model = os.path.abspath('models/retinaface-R50/R50')
    face_detector = RetinaFace(retina_face_model, 0, 0, 'net3')

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def process_images(self, image_path):
        face_list = []
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

                # append to list
                x1,y1,x2,y2 = box[0], box[1], box[2], box[3]
                ch = y2 - y1
                cw = x2 - x1
                org_img = img.copy()
                crop_img = org_img[y1:y1 + ch, x1:x1 + cw]
                face_list.append(crop_img)

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
        cv2.imshow('Images', img)

        # Hit 'q' on the keyboard to quit!
        cv2.waitKey(0)

        return face_list

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


class FaceRecognizer:
    arc_face_model = os.path.abspath('models/model-r100-arcface-ms1m-refine-v2/model-r100-ii/model-0000')
    recognizer = FaceRecognition(arc_face_model)

    def __init__(self):
        self.recognizer.prepare(0)

    def process_img(self, img):
        emb1 = self.recognizer.get_embedding(img)
        print(emb1)

        emb2 = self.recognizer.get_embedding(img)
        print(emb2)

        score = self.recognizer.compute_sim(img, img)
        print(score)


@timeit
def recognize_images():
    # # read image from file
    # img = cv2.imread('test_data/rahat1.jpg')
    # img = imutils.resize(img, 112, 112)
    # fr = FaceRecognizer()
    # fr.process_img(img)
    # return
    with FaceDetector() as face_rec:
        test_images = []
        for f in glob.glob('test_data/*.jpg'):
            test_images.append(f)

        for im in test_images:
            faces = face_rec.process_images(im)
            for f in faces:
                cv2.imshow('Crop Image', f)
                # Hit 'q' on the keyboard to quit!
                cv2.waitKey(0)
            break

        # with ThreadPool(1) as pool:
        #     pool.map(face_rec.process_images, test_images)

    cv2.destroyAllWindows()


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
