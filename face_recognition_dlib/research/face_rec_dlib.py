import glob

import cv2
import dlib
import numpy as np


class FaceRecognizer:
    PREDICTOR_PATH = 'ml_models/shape_predictor_5_face_landmarks.dat'
    FACE_REC_MODEL_PATH = 'ml_models/dlib_face_recognition_resnet_model_v1.dat'
    known_face_encodings = []

    def __init__(self):
        dlib.DLIB_USE_CUDA = True

    def __enter__(self):
        self.__init_dlib()
        self.__load_face_encodings()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def __init_dlib(self):
        # Load all the models we need: a detector to find the faces, a shape predictor
        # to find face landmarks so we can precisely localize the face, and finally the
        # face recognition model.
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(self.PREDICTOR_PATH)
        self.facerec = dlib.face_recognition_model_v1(self.FACE_REC_MODEL_PATH)

    def __load_face_encodings(self):
        for img in glob.glob('./train_data/*.jpg'):
            # Load a sample picture and learn how to recognize it.
            train_image = dlib.load_rgb_image(img)

            dets = self.detector(train_image, 1)
            print("Number of faces detected: {}".format(len(dets)))

            # Get the landmarks/parts for the face in box d.
            shape = self.sp(train_image, dets[0])

            # Let's generate the aligned image using get_face_chip
            face_chip = dlib.get_face_chip(train_image, shape)

            # Now we simply pass this chip (aligned image) to the api
            descriptor_prealigned_image = self.facerec.compute_face_descriptor(face_chip)

            self.known_face_encodings.append((np.array(descriptor_prealigned_image), img.split('/')[-1].replace('.jpg', '')))

    def __extract_landmarks(self, img):
        pass

    def __calculate_face_distance(self, face_encodings, face_to_compare):
        return np.linalg.norm(face_encodings - face_to_compare)

    def process_image(self, image_path):
        # read image with opencv
        img = cv2.imread(image_path)

        # convert it to rgb
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # detect all faces
        dets = self.detector(rgb_img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        # Now process each face we found.
        for k, d in enumerate(dets):
            left, top, right, bottom = d.left(), d.top(), d.right(), d.bottom()
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(),
                                                                               d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = self.sp(img, d)

            # Let's generate the aligned image using get_face_chip
            face_chip = dlib.get_face_chip(rgb_img, shape)

            # Now we simply pass this chip (aligned image) to the api
            face_descriptor_from_prealigned_image = self.facerec.compute_face_descriptor(face_chip)
            # print(face_descriptor_from_prealigned_image)

            # Draw a box around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)


            # print(self.known_face_encodings)
            if(len(face_descriptor_from_prealigned_image) == 0):
                continue

            distances = [(self.__calculate_face_distance(train_data[0], np.array(face_descriptor_from_prealigned_image)), train_data[1]) for train_data in self.known_face_encodings]
            index = np.argmin([d[0] for d in distances])

            # Draw a label with a name below the face
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX

            dis = distances[index]
            text = '{}\n{}'.format(dis[1], dis[0])

            cv2.putText(img, text, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Mehmet', img)


        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    with FaceRecognizer() as face_rec:
        for f in glob.glob('./test_data/*.jpg'):
            face_rec.process_image(f)
