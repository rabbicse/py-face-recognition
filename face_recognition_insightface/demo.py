import insightface
import urllib
import urllib.request
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


# url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
# img = url_to_image(url)

img = cv2.imread('test_data/3sis.jpg')
model = insightface.app.FaceAnalysis()
ctx_id = -1
model.prepare(ctx_id=ctx_id, nms=0.4)
faces = model.get(img)
for idx, face in enumerate(faces):
    bbox = face.bbox.astype(np.int)
    print(face.bbox[0])
    print("Face [%d]:" % idx)
    print("\tage:%d" % (face.age))
    gender = 'Male'
    if face.gender == 0:
        gender = 'Female'
    print("\tgender:%s" % (gender))
    print("\tembedding shape:%s" % face.embedding.shape)
    print("\tbbox:%s" % (face.bbox.astype(np.int).flatten()))
    print("\tlandmark:%s" % (face.landmark.astype(np.int).flatten()))
    print("")

    # Draw a box around the face
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

    text = 'Age: {}'.format(face.age)
    # text = 'Gender: {}'.format(gender)
    # print(text)
    size = cv2.getTextSize(text, FONT, 0.55, 2)
    # Draw a label with a name below the face
    # cv2.rectangle(img, (left, bottom), (left + size[0][0] + 6, bottom + size[0][1] + 6), (0, 0, 255),
    #               cv2.FILLED)

    # cv2.putText(img, text, (bbox[0] + 6, bbox[3] + size[0][1]), FONT, 0.55, (255, 255, 255), 1)
    # Draw a label with a name below the face

    left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    cv2.rectangle(img, (left, bottom + 8), (left + size[0][0] + 8, bottom + size[0][1] + 20), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, text, (left + 5, bottom + size[0][1] + 12), FONT, 0.55, (255, 255, 255), 1)

cv2.imwrite('sisters_bak.jpg', img)
cv2.imshow('Face Age', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
