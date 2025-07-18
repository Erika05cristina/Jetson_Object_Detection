# processor/main.py
import numpy as nu
import logging as log 

from processor.detection.object_detector import ObjecDetectionInterface, ObjectDetection
from processor.draw.main import DrawingInterface, Drawing 

class ObjectProcessing:
    def _init_(self):
        self.detection: ObjecDetectionInterface = ObjectDetection()
        self.draw_detection: DrawingInterface = Drawing()

    def frame_processing(self, image:np.ndarray) -> np.ndarray
        """
        """
        log.info("Processing frame for object detection.")
        try:
            inference_image = image.copy()
            
            #Deteccion
            detection_results, detection_classes = self.detection.inference(inference_image)
            log.info(f"Detected {len(detection_results)} objects.")

            # Draw detection
            for objects in detection_results:
                image_draw = image.copy()
                image_draw = self.draw_detection.draw(image_draw, objects, detection_classes)
            log.info("Drawing completed. ")
            return image_draw





        except Exception as e:
            log.error(f"Error during object detection: {e}")
            return image  # Return original image if detection fails
      
        
        return processed_image
