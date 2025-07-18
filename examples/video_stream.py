# examples/video_stream.py
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processor.main import ObjectProcessing

class VideoStream:
    def __init__(self, video_url="/home/je/Desktop/Proyecto/YoloJN/examples/video.mp4"):
        self.object_detector = ObjectProcessing()
        self.cap = cv2.VideoCapture(video_url)
        if not self.cap.isOpened():
            raise ValueError("No se pudo abrir la fuente de video")

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("No se pudo leer el frame.")
                break

            frame_processed = self.object_detector.frame_processing(frame)

            cv2.imshow('Video desde archivo', frame_processed)
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
