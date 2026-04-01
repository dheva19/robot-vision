from flask import Flask, Response, jsonify
from ultralytics import YOLO
import cv2
import time
import random

app = Flask(__name__)

model = YOLO('ms_bima.pt')
print("Model loaded successfully")

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Kamera tidak ditemukan")
        return

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        results = model(frame, imgsz=128, conf=0.50, verbose=False)
        
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0 
            start_time = time.time()
            
        annotated_frame = results[0].plot()
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            text_coord = f"X: {int(x1)} Y: {int(y1)}"
            cv2.putText(
                annotated_frame, 
                text_coord, 
                (int(x1), int(y1) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 0, 0), 
                1
            )

        num_objects = len(results[0].boxes)
        cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Objects: {num_objects}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return "Video Streaming Running at /video_feed"

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sensor_data')
def sensor_data():
    data = {
        "ultrasonic": {
            "sensor_1": round(random.uniform(10.0, 100.0), 2),
            "sensor_2": round(random.uniform(10.0, 100.0), 2),
            "sensor_3": round(random.uniform(10.0, 100.0), 2),
            "sensor_4": round(random.uniform(10.0, 100.0), 2),
            "sensor_5": round(random.uniform(10.0, 100.0), 2),
            "sensor_6": round(random.uniform(10.0, 100.0), 2)
        },
        "imu": {
            "yaw": round(random.uniform(-180.0, 180.0), 2),
            "pitch": round(random.uniform(-90.0, 90.0), 2),
            "roll": round(random.uniform(-180.0, 180.0), 2)
        },
        "timestamp": time.time()
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)