# train.py
import cv2
import os
import numpy as np

DATASET_DIR = "data sets"
MODEL_FILE = "face_trainer.yml"

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = []
    labels = []
    label_map = {}
    current_id = 0

    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_map[current_id] = person_name

        for img_name in os.listdir(person_dir):
            path = os.path.join(person_dir, img_name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces_detected = face_cascade.detectMultiScale(img, 1.1, 4)
            for (x, y, w, h) in faces_detected:
                face_roi = img[y:y+h, x:x+w]
                faces.append(face_roi)
                labels.append(current_id)

        current_id += 1

    if not faces:
        print("No faces found in dataset. Add images first.")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_FILE)

    # save label map
    np.save("labels.npy", label_map)

    print(f"Training complete. Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_model()
