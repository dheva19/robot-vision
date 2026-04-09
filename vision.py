from ultralytics import YOLO
import cv2
import time

def main():
    model = YOLO('ms_bima.pt')
    print("Model loaded successfully")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam")
        return
    
    print("Memulai inferensi... Tekan 'q' atau Ctrl+C untuk berhenti")

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame")
                break

            results = model(
                frame,
                imgsz=128,
                conf=0.50,
                verbose=False
            )

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
                print(f"x : {int(x1)} | y: {int(x2)}")
                cv2.putText(
                    annotated_frame, 
                    text_coord, 
                    (int(x1), int(y1) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 0, 0), 
                    1
                )

            cv2.putText(
                annotated_frame,
                f'FPS: {fps:.2f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            num_objects = len(results[0].boxes)
            cv2.putText(
                annotated_frame,
                f'Objects: {num_objects}',
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            # print(f"FPS : {fps}")

            cv2.imshow('YOLO Realtime Detection', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nMenghentikan inferensi...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam ditutup")

if __name__ == "__main__":
    main()