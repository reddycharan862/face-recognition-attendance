from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from PIL import Image
import io
import csv
import os
from datetime import datetime, date

app = Flask(__name__)

# Load model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_trainer.yml")

label_map = np.load("labels.npy", allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

ATTENDANCE_FILE = "attendance.csv"

# create CSV if not exists
def ensure_csv():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

def already_marked(name):
    today = date.today().isoformat()
    if not os.path.exists(ATTENDANCE_FILE):
        return False
    with open(ATTENDANCE_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Name"] == name and row["Date"] == today:
                return True
    return False

def mark_attendance(name):
    now = datetime.now()
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, now.date(), now.strftime("%H:%M:%S")])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    ensure_csv()

    data = request.get_json()
    image_data = data["image"]

    image_data = image_data.split(",")[1]
    img_bytes = base64.b64decode(image_data)

    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame = np.array(image)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return jsonify({"message": "No face detected"})

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi)

        if conf < 70:
            name = label_map[id_]

            if already_marked(name):
                return jsonify({"message": f"{name} already marked today"})
            else:
                mark_attendance(name)
                return jsonify({"message": f"Attendance marked for {name}"})

    return jsonify({"message": "Could not recognize face"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)