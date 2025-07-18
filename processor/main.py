# processor/main.py
import numpy as np
import logging as log
import cv2
import torch

from processor.detection.object_detector import ObjecDetectionInterface, ObjectDetection
from processor.draw.main import DrawingInterface, Drawing

class ObjectProcessing:
    def __init__(self):
        self.detection: ObjecDetectionInterface = ObjectDetection()
        self.draw_detection: DrawingInterface = Drawing()

    def frame_processing(self, image: np.ndarray) -> np.ndarray:
        log.info("Processing frame for object detection.")
        try:
            inference_image = image.copy()

            detection_results, detection_classes = self.detection.inference(inference_image)
            log.info(f"Detected {len(detection_results)} objects.")

            image_draw = image.copy()
            for objects in detection_results:
                image_draw = self.draw_detection.draw(image_draw, objects, detection_classes)

            # ✅ Overlay: texto indicando device CUDA/CPU en la imagen:
            device_label = "Device: CUDA" if torch.cuda.is_available() else "Device: CPU"
            color = (0, 255, 0) if torch.cuda.is_available() else (0, 0, 255)
            cv2.putText(
                image_draw,
                device_label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                cv2.LINE_AA
            )

            log.info("Drawing completed.")
            return image_draw

        except Exception as e:
            log.error(f"Error during object detection: {e}")
            return image
