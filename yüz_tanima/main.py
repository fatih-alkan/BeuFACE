import cv2
import dlib
import numpy as np
import os
from tkinter import Tk, Button, Label, Entry, StringVar

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

known_faces = []
known_names = []

def load_known_faces():
    known_faces.clear()
    known_names.clear()
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')
    for file in os.listdir('known_faces'):
        image = cv2.imread(f'known_faces/{file}')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            shape = sp(gray, faces[0])
            face_descriptor = facerec.compute_face_descriptor(image, shape)
            known_faces.append(np.array(face_descriptor))
            known_names.append(file.split('.')[0])
    print("Bilinen yüzler yüklendi: ", known_names)

def save_face(image, name):
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')
    cv2.imwrite(f'known_faces/{name}.jpg', image)

def capture_and_save(name):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Kamera açılamadı!")
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        save_face(frame, name)
        load_known_faces()
        print(f"{name} kaydedildi.")
    else:
        print("Yüz algılanamadı.")
    cv2.imshow("Yüz", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release() 

def recognize_face():
    load_known_faces()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Kamera açılamadı!")
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        for face in faces:
            shape = sp(gray, face)
            face_descriptor = facerec.compute_face_descriptor(frame, shape)
            face_descriptor = np.array(face_descriptor)
            if len(known_faces) > 0:
                matches = np.linalg.norm(known_faces - face_descriptor, axis=1)
                min_index = np.argmin(matches)
                if matches[min_index] < 0.6:
                    name = known_names[min_index]
                    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    load_known_faces()
                    print(f"{name} tanındı.")
                    
                else:
                    name = "Bilinmeyen"
                    print("Eşleşme bulunamadı.")
                    print(f"Tanınan kişi: {name}")
            else:
                print("Bilinen yüzler listesi boş.")
    else:
        print("Yüz algılanamadı.")
    cv2.imshow("Yüz", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

def on_capture_button_click():
    capture_window = Tk()
    capture_window.title("İsim Girin")

    label = Label(capture_window, text="İsim girin:")
    label.pack()

    name_var = StringVar()
    name_entry = Entry(capture_window, textvariable=name_var)
    name_entry.pack()

    def save_and_close():
        name = name_entry.get().strip()
        print("Girilen İsim: ", name)
        if name: 
            capture_and_save(name)
        else:
            print("İsim boş olamaz!")
        capture_window.destroy()

    save_button = Button(capture_window, text="Kaydet", command=save_and_close)
    save_button.pack()

    capture_window.mainloop()

root = Tk()
root.title("Yüz Tanıma")

capture_button = Button(root, text="Yüzü Kaydet", command=on_capture_button_click)
capture_button.pack()

recognize_button = Button(root, text="Yüzü Tanı", command=recognize_face)
recognize_button.pack()

load_known_faces()
root.mainloop()
