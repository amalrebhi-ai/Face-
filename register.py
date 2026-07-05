import cv2
import face_recognition
import numpy as np
from db import add_person

name = input("Nom : ")
info = input("Info (séparé par ,) : ").split(",")

cap = cv2.VideoCapture(0)

print("📸 Regarde la caméra...")

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    encodings = face_recognition.face_encodings(rgb)

    cv2.imshow("Register", frame)

    if len(encodings) > 0:
        encoding = encodings[0]

        print("✔ Visage capturé")

        add_person(name, info, encoding)

        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()