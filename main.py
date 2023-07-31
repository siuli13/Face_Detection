import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime


# Function to find face encodings for known images in the specified folder
def findEncodings(images_folder):
    encodeList = []
    classNames = []
    myList = os.listdir(images_folder)
    for cl in myList:
        curImg = cv2.imread(f"{images_folder}/{cl}")
        classNames.append(os.path.splitext(cl)[0])
        curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(curImg)[0]
        encodeList.append(encode)
    return encodeList, classNames


# Function to mark attendance in the CSV file
def markAttendance(name):
    with open("Attendance.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = [line.split(",")[0] for line in myDataList]
        if name not in nameList:
            time_now = datetime.now()
            tString = time_now.strftime("%H:%M:%S")
            dString = time_now.strftime("%d/%m/%Y")
            f.writelines(f"\n{name},{tString},{dString}")


# Loading known face encodings and class names
encodeListKnown, classNames = findEncodings("Images_Attendance")
print("Encoding Complete")

# Setting up the webcam for live face recognition
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set Width
cap.set(4, 480)  # Set Height

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, +1)  # Flip camera vertically

    # Face recognition on the live webcam frame
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 250, 0), cv2.FILLED)
            cv2.putText(
                frame,
                name,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            markAttendance(name)

    cv2.imshow("webcam", frame)
    if cv2.waitKey(10) == 27:  # Press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()

# The rest of the code for comparing two images
imgModi = face_recognition.load_image_file("Images_Attendance/Modi.jpg")
imgModi = cv2.cvtColor(imgModi, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("Images_Attendance/Modi2.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgModi)[0]
encodeModi = face_recognition.face_encodings(imgModi)[0]
cv2.rectangle(
    imgModi, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (155, 0, 255), 2
)

facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[1]
cv2.rectangle(
    imgTest,
    (facelocTest[3], facelocTest[0]),
    (facelocTest[1], facelocTest[2]),
    (155, 0, 255),
    2,
)

results = face_recognition.compare_faces([encodeModi], encodeTest)
faceDis = face_recognition.face_distance([encodeModi], encodeTest)
print(results, faceDis)
cv2.putText(
    imgTest,
    f"{results} {round(faceDis[0],2)}",
    (50, 50),
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    1,
    (0, 0, 255),
    2,
)

cv2.imshow("modi", imgModi)
cv2.imshow("narendra-modi", imgTest)
cv2.waitKey(0)
cv2.destroyAllWindows()
