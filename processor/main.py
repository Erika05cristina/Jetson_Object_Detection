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

            # ✅ Eliminado: No mostrar texto del dispositivo
            log.info("Drawing completed.")
            return image_draw

        except Exception as e:
            log.error(f"Error during object detection: {e}")
            return image
