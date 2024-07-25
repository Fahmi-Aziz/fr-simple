import cv2
import numpy as np
import face_recognition
import os

# path folder foto
path = "faces"

# load gambar
classNames = [os.path.splitext(cl)[0] for cl in os.listdir(path)]
images = [cv2.imread(os.path.join(path, cl)) for cl in os.listdir(path)]

# encode 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

# pilih sumber gambar 
cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Mendeteksi wajah
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # Membandingkan wajah dgn yg sudah di encode
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].capitalize()
            top, right, bottom, left = [i * 4 for i in faceLoc]

            # buat rectangle
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

            # buat nama class sesuai nama gambar 
            cv2.putText(img, name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Deteksi Orang', img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
cam.release()
