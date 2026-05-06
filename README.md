Face Recognition Attendance System
A web-based face recognition attendance system built with Python, Flask, and OpenCV. It captures an image from the browser webcam, detects the face, recognizes the person using a trained model, and records attendance automatically.

Features

Face detection using webcam
Face recognition using LBPH algorithm
Attendance stored in CSV file
Prevents duplicate attendance for the same day
Simple web interface using Flask


Technologies Used

Python
Flask
OpenCV
NumPy
HTML, CSS, JavaScript(basic frontend)


Project Structure
face_attendance/
app.py                  # Main Flask application
face_trainer.yml        # Trained LBPH face recognition model
labels.npy              # Label mappings (ID to Name)
requirements.txt        # Python dependencies
Procfile                # Deployment startup command
templates/
    index.html          # Frontend web interface


How to Run

Clone the repository => git clone https://github.com/reddycharan862/face_attendance.git
cd face attendance

Install required packages => pip install -r requirements.txt

Run the application => python app.py

Open your browser and go to => http://127.0.0.1:5000

How It Works

1 . The browser accesses the webcam using JavaScript
2 . The user clicks the Mark Attendance button
3 . A snapshot is captured and sent to the Flask backend
4 . OpenCV detects the face and the LBPH model recognizes the person
5 . Attendance is recorded in a CSV file with name, date, and time
6 . A message is displayed confirming whether attendance was marked or already exists


Output

=> The system detects and recognizes the face from webcam input
=> Attendance is recorded in a CSV file
=> Messages are displayed indicating whether attendance is marked or not


Deployment
This application is deployed using Render with Flask and Gunicorn.
Procfile => web: gunicorn app:app


Render Settings:
Environment - Python
Build Command - pip install -r requirements.txt
Start Command - gunicorn app:app
