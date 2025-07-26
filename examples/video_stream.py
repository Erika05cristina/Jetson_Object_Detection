# examples/video_stream.py
import os
import sys
import cv2
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processor.main import ObjectProcessing

class VideoStream:
    def __init__(self, video_url="http://192.168.83.169:81/stream"):
        self.object_detector = ObjectProcessing()
        self.cap = cv2.VideoCapture(video_url)
        if not self.cap.isOpened():
            raise ValueError("No se pudo abrir la fuente de video")

    def run(self):
        prev_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("No se pudo leer el frame.")
                break

            # Calcular FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            # Procesar el frame
            frame_processed = self.object_detector.frame_processing(frame)

            # Dibujar FPS en el frame
            cv2.putText(
                frame_processed,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Mostrar el frame
            cv2.imshow('Video desde ESP32-CAM', frame_processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    stream = VideoStream()
    try:
        stream.run()
    except KeyboardInterrupt:
        print("Stream interrumpido por el usuario.")
        stream.cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)
