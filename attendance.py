# attendance.py
import cv2
import numpy as np
import os
import csv
from datetime import datetime, date

MODEL_FILE = "face_trainer.yml"
LABELS_FILE = "labels.npy"
ATTENDANCE_FILE = "attendance.csv"

def ensure_csv_exists():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

def already_marked_today(name):
    today_str = date.today().isoformat()
    if not os.path.exists(ATTENDANCE_FILE):
        return False
    with open(ATTENDANCE_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Name"] == name and row["Date"] == today_str:
                return True
    return False

def mark_attendance(name):
    now = datetime.now()
    d = now.date().isoformat()
    t = now.strftime("%H:%M:%S")
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, d, t])
    print(f"Marked attendance for {name} at {t}")

def run_attendance():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
        print("Train the model first using train.py")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    label_map = np.load(LABELS_FILE, allow_pickle=True).item()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    ensure_csv_exists()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi_gray)

            if conf < 70:  # confidence threshold
                name = label_map[id_]
                if not already_marked_today(name):
                    mark_attendance(name)
            else:
                name = "Unknown"

            # draw box + name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_attendance()
