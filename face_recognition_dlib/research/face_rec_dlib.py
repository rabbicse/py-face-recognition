import glob
import time
from multiprocessing.pool import ThreadPool

import cv2
import dlib
import dlib.cuda as cuda
import imutils
import numpy as np

SP_PREDICTOR_5_POINT_PATH = 'ml_models/shape_predictor_5_face_landmarks.dat'
SP_PREDICTOR_68_POINT_PATH = 'ml_models/shape_predictor_68_face_landmarks.dat'
FACE_REC_MODEL_PATH = 'ml_models/dlib_face_recognition_resnet_model_v1.dat'
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


class FaceRecognizer:
    __shape_predictor_5_point = dlib.shape_predictor(SP_PREDICTOR_5_POINT_PATH)
    __shape_predictor_68_point = dlib.shape_predictor(SP_PREDICTOR_68_POINT_PATH)
    __face_rec_model = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)
    __detector = dlib.get_frontal_face_detector()
    known_face_encodings = []

    def __init__(self):
        print(dlib.DLIB_USE_CUDA)
        print(cuda.get_num_devices())

    def __enter__(self):
        self.__load_face_encodings()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def __load_face_encodings(self):
        for img in glob.glob('./train_data/*.jpg'):
            # Load a sample picture and learn how to recognize it.
            train_image = dlib.load_rgb_image(img)

            # detect all faces
            detected_faces = self.__detector(train_image, 1)
            # print("Number of faces detected: {}".format(len(dets)))

            if len(detected_faces) == 0:
                continue

            # Extract landmarks for all trainning images
            face_chip = self.__extract_landmarks(train_image, detected_faces[0])

            # Now we simply pass this chip (aligned image) to the api
            descriptor_prealigned_image = self.__face_rec_model.compute_face_descriptor(face_chip)

            self.known_face_encodings.append(
                (np.array(descriptor_prealigned_image), img.split('/')[-1].replace('.jpg', '')))

    def __extract_landmarks(self, train_image, detected_face):
        # Get the landmarks/parts for the face in box d.
        shape = self.__shape_predictor_5_point(train_image, detected_face)

        # Let's generate the aligned image using get_face_chip
        face_chip = dlib.get_face_chip(train_image, shape)

        return face_chip

    def __calculate_face_distance(self, face_encodings, face_to_compare):
        return np.linalg.norm(face_encodings - face_to_compare)

    def process_images(self, image_path):
        # read image with opencv
        img = cv2.imread(image_path)

        # get image height and width
        (h, w) = img.shape[:2]

        # resize image
        if w > 720 or h > 720:
            img = imutils.resize(img, width=720) if w > h else imutils.resize(img, height=720)

        # convert it to rgb
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.process_rgb_image(img, rgb_img)

        cv2.imwrite('output_data/{}'.format(image_path.split('/')[-1]), img)

    def process_frames(self, img):
        # get image height and width
        (h, w) = img.shape[:2]

        # resize image
        if w > 720 or h > 720:
            img = imutils.resize(img, width=720) if w > h else imutils.resize(img, height=720)

        # convert it to rgb
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.process_rgb_image(img, rgb_img)

        # cv2.imwrite('output_data/{}'.format(image_path.split('/')[-1]), img)

        cv2.imshow('Video', img)

    # @timeit
    def process_rgb_image(self, img, rgb_img):
        # detect all faces
        detected_faces = self.__detector(rgb_img, 1)
        print("Number of faces detected: {}".format(len(detected_faces)))

        # Now process each face we found.
        for k, d in enumerate(detected_faces):
            left, top, right, bottom = d.left(), d.top(), d.right(), d.bottom()

            face_chip = self.__extract_landmarks(rgb_img, d)

            # Now we simply pass this chip (aligned image) to the api
            face_descriptor_from_prealigned_image = self.__face_rec_model.compute_face_descriptor(face_chip)
            # print(face_descriptor_from_prealigned_image)

            # Draw a box around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

            # print(self.known_face_encodings)
            if len(face_descriptor_from_prealigned_image) == 0:
                continue

            distances = [(
                self.__calculate_face_distance(train_data[0], np.array(face_descriptor_from_prealigned_image)),
                train_data[1]) for train_data in self.known_face_encodings]
            index = np.argmin([d[0] for d in distances])
            dis = distances[index]

            if dis[0] >= 0.5:
                continue

            text = '{} : {} %'.format(dis[1], round((100 - dis[0] * 100), 1))
            print(text)
            size = cv2.getTextSize(text, FONT, 0.55, 2)
            # Draw a label with a name below the face
            cv2.rectangle(img, (left, bottom), (left + size[0][0] + 6, bottom + size[0][1] + 6), (0, 0, 255),
                          cv2.FILLED)

            cv2.putText(img, text, (left + 6, bottom + size[0][1]), FONT, 0.55, (255, 255, 255), 1)

        # cv2.imshow('Mehmet', img)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # cv2.imwrite('output_data/{}'.format(image_path.split('/')[-1]), img)


@timeit
def recognize_images():
    with FaceRecognizer() as face_rec:
        test_images = []
        for f in glob.glob('./test_data/*.jpg'):
            test_images.append(f)

        # for im in test_images:
        #     face_rec.process_image(im)

        # for i in range(10):
        with ThreadPool(32) as pool:
            pool.map(face_rec.process_images, test_images)


def recognize_from_webcam():
    webcam = cv2.VideoCapture(0)
    with FaceRecognizer() as face_rec:
        while True:
            # Grab a single frame of video
            ret, frame = webcam.read()

            face_rec.process_frames(frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    # recognize_images()
    recognize_from_webcam()
