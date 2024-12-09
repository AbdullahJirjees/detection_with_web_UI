import cv2
from ultralytics import YOLO
import cvzone
import pandas as pd
import time
from flask import Flask, Response, render_template, jsonify
import json

# Load model and class names
model = YOLO("/home/abdullah/Projects/yolov11_testing/weights/yolo11n.pt")  # Adjust path if needed
classNames = ['person', 'car']

# Initialize Flask app
app = Flask(__name__)

def generate_frames():
    # Open video source (adjust path or use webcam as needed)
    cap = cv2.VideoCapture(0)
    # Create DataFrame for storing detections
    df = pd.DataFrame(columns=['Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
    # Initialize variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        new_frame_time = time.time()

        # Capture frame
        success, img = cap.read()
        if not success:
            print("Error reading frame from video stream.")
            break

        # Perform object detection
        results = model(img, stream=True)

        for r in results:
            for box in r.boxes:
                # Extract bounding box coordinates, confidence, and class
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                try:
                    # Handle potential class index out of range
                    if 0 <= cls < len(classNames):
                        # Draw bounding box and label using cvzone
                        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1))
                        cvzone.putTextRect(img, f'{classNames[cls]} {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                    else:
                        print(f"Warning: Class index {cls} out of range. Using default label.")
                        cvzone.putTextRect(img, "Unknown Class", (max(0, x1), max(35, y1)), scale=1, thickness=1)

                    # Add detection to DataFrame
                    df = pd.concat([df, pd.DataFrame({'Class': classNames[cls], 'Confidence': conf, 'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2}, index=[0])], ignore_index=True)
                except IndexError:
                    print("Error: Index out of range. Skipping detection.")

        # Encode the frame in JPEG Format:
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        # Frame to flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    print("Video capture released for video stream.")

@app.route('/')
def index():
    # Render the main page
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_status')
def detection_status():
    def detect():
        cap = cv2.VideoCapture(0)  # Open video capture for detection
        while True:
            success, img = cap.read()
            if not success:
                yield 'data: {"status": "error"}\n\n'
                continue

            # Perform object detection
            results = model(img, stream=True)
            person_detected = False
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == classNames.index('person'):
                        person_detected = True
                        break

            # Send detection status
            status = {"status": "detected" if person_detected else "clear"}
            yield f'data: {json.dumps(status)}\n\n'

        cap.release()
        print("Video capture released for detection.")

    return Response(detect(), content_type='text/event-stream')

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)
