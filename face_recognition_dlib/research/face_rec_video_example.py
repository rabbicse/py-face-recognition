import glob

import PIL
import cv2
import dlib
import face_recognition
import numpy as np


face_recognition_model = './ml_models/dlib_face_recognition_resnet_model_v1.dat'
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
pose_predictor_68_point = dlib.shape_predictor('./ml_models/shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()

def load_image_file(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)


def face_encodingsv1(face_image, known_face_locations=None, num_jitters=1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model="small")
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    # if face_locations is None:
    # face_locations = _raw_face_locations(face_image)
    # else:
    #     face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    # pose_predictor = pose_predictor_68_point
    #
    # if model == "small":
    #     pose_predictor = pose_predictor_5_point

    face_locations = _raw_face_locations(face_image)


    return [pose_predictor_68_point(face_image, face_location) for face_location in face_locations]

def _raw_face_locations(img, number_of_times_to_upsample=1, model="hog"):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of dlib 'rect' objects of found face locations
    """
    if model == "cnn":
        cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
        return cnn_face_detector(img, number_of_times_to_upsample)
    else:
        # face_detector = dlib.get_frontal_face_detector()
        return face_detector(img, number_of_times_to_upsample)

class Webcam:
    known_face_encodings = []

    def __init__(self):
        dlib.DLIB_USE_CUDA = True
        self.__webcam = cv2.VideoCapture(0)

    def __enter__(self):
        self.__load_face_encodings()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Release handle to the webcam
        self.__webcam.release()
        cv2.destroyAllWindows()
        del self.__webcam

    def __load_face_encodings(self):
        for img in glob.glob('./train_data/*.jpg'):
            # Load a sample picture and learn how to recognize it.
            train_image = load_image_file(img)
            train_face_encoding = face_encodingsv1(train_image)[0]
            self.known_face_encodings.append((train_face_encoding, img.split('/')[-1].replace('.jpg', '')))

    def process_frame(self):
        process_this_frame = True
        known_faces_encodings = [en[0] for en in self.known_face_encodings]
        known_faces_names = [en[1] for en in self.known_face_encodings]
        while True:
            # Grab a single frame of video
            ret, frame = self.__webcam.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            # rgb_small_frame = frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_encodingsv1(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
                    print(face_distances)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_faces_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                # top *= 4
                # right *= 4
                # bottom *= 4
                # left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            # cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    with Webcam() as webcam:
        webcam.process_frame()
