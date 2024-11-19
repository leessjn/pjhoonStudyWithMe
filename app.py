from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import json
import os
import threading
import time

app = Flask(__name__)
CORS(app)

class StudyMonitor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.rankings_file = 'study_rankings.json'
        self.rankings = self.load_rankings()
        self.current_user = None
        self.total_study_time = 0
        self.session_start_time = None
        self.is_studying = False
        self.capture = None

    def load_rankings(self):
        if not os.path.exists(self.rankings_file):
            return []
        with open(self.rankings_file, 'r') as f:
            return json.load(f)

    def save_rankings(self):
        with open(self.rankings_file, 'w') as f:
            json.dump(self.rankings, f)

    def update_rankings(self, username, study_time):
        for user in self.rankings:
            if user['name'] == username:
                user['total_study_time'] += study_time
                break
        else:
            self.rankings.append({
                'name': username,
                'total_study_time': study_time
            })
        
        self.rankings.sort(key=lambda x: x['total_study_time'], reverse=True)
        self.save_rankings()

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0

    def generate_frames(self):
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                break

            face_detected = self.detect_face(frame)

            if face_detected and self.is_studying:
                current_time = time.time()
                if self.session_start_time is None:
                    self.session_start_time = current_time
                
                session_duration = int(current_time - self.session_start_time)
                cv2.putText(frame, f"Studying: {session_duration}s", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def start_study_session(self, username):
        self.capture = cv2.VideoCapture(0)
        self.current_user = username
        self.is_studying = True
        self.session_start_time = None

    def stop_study_session(self):
        if self.session_start_time:
            study_duration = int(time.time() - self.session_start_time)
            self.update_rankings(self.current_user, study_duration)
            self.total_study_time += study_duration
        
        self.is_studying = False
        self.session_start_time = None
        self.capture.release()
        self.capture = None

monitor = StudyMonitor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(monitor.generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_study', methods=['POST'])
def start_study():
    username = request.json.get('username', 'Anonymous')
    monitor.start_study_session(username)
    return jsonify({"status": "Study session started"})

@app.route('/stop_study', methods=['POST'])
def stop_study():
    monitor.stop_study_session()
    return jsonify({
        "status": "Study session stopped", 
        "rankings": monitor.rankings
    })

if __name__ == '__main__':
    app.run(debug=True)
