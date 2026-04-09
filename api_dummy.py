from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import time
import random

app = Flask(__name__)
# Izinkan akses dari Laravel React
CORS(app, origins=["http://127.0.0.1:8000", "http://localhost:8000"])

# --- State Global untuk Data Dummy Stabil ---
robot_state = {
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0
}

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
        
        # imgsz kecil untuk performa, conf 0.50 sesuai permintaan
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
    global robot_state
    
    # 1. Update IMU dengan Random Walk (Perubahan kecil agar stabil)
    # Yaw bisa berputar bebas, tapi Pitch & Roll biasanya stabil pada robot SAR
    robot_state["yaw"] += random.uniform(-1.0, 1.0)
    robot_state["pitch"] += random.uniform(-0.3, 0.3)
    robot_state["roll"] += random.uniform(-0.3, 0.3)

    # 2. Batasi (Clamping) agar tidak miring berlebihan (Robot tidak terbalik)
    robot_state["pitch"] = max(-10.0, min(10.0, robot_state["pitch"]))
    robot_state["roll"] = max(-10.0, min(10.0, robot_state["roll"]))
    
    # Reset Yaw jika melebihi putaran penuh
    if abs(robot_state["yaw"]) > 180:
        robot_state["yaw"] = -180 if robot_state["yaw"] > 0 else 180

    data = {
        "ultrasonic": {
            "sensor_1": round(random.uniform(30.0, 60.0), 2),
            "sensor_2": round(random.uniform(30.0, 60.0), 2),
            "sensor_3": round(random.uniform(10.0, 30.0), 2), # Simulasi rintangan dekat
            "sensor_4": round(random.uniform(80.0, 100.0), 2),
            "sensor_5": round(random.uniform(40.0, 50.0), 2),
            "sensor_6": round(random.uniform(90.0, 100.0), 2)
        },
        "imu": {
            "yaw": round(robot_state["yaw"], 2),
            "pitch": round(robot_state["pitch"], 2),
            "roll": round(robot_state["roll"], 2)
        },
        "timestamp": time.time()
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)