from flask import Flask, Response, render_template
from ultralytics import YOLO
import cv2
import time

app = Flask(__name__)

# Load Model
model = YOLO('best.pt')

def gen_frames():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    start_time = time.time()
    fps = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Inferensi YOLO
        results = model(frame, imgsz=128, conf=0.50, verbose=False)
        
        # Kalkulasi FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        # Plot hasil deteksi ke frame
        annotated_frame = results[0].plot()
        
        # Tambah Text FPS & Object Count
        num_objects = len(results[0].boxes)
        cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Objects: {num_objects}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        cv2.imshow("test", annotated_frame)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # Mengirim file HTML utama
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Endpoint untuk stream video
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Host 0.0.0.0 agar bisa diakses dari perangkat lain dalam satu WiFi
    app.run(host='0.0.0.0', port=5000, debug=False)