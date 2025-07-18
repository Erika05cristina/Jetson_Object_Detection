#examples/video_stream.py
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__), "..")))

from processor.main import ObjectProcessing

class video_stream:
    def __init__(self, video_source=0):

        self.object_detector = ObjectProcessing()
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError("Could not open video source")

    def run(self):
        while True:
            ret, frame = self.cap.read()
            frame_processed = self.object_detector.frame_processing(frame)

            if not ret:
                break
                       
            cv2.imshow('Video Stream', frame_processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_stream = VideoStream()
    try:
        video_stream.run()
    except KeyboardInterrupt:
        print("Video stream interrupted by user.")
    
        video_stream.cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)
