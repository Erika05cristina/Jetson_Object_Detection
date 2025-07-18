# processor/detection/object_detector.py
import os
import logging as log
import torch
import numpy as np

from ultralytics import YOLO
from abc import ABC, abstractmethod
from ultralytics.engine.results import Results

class ObjecDetectionInterface(ABC):
    @abstractmethod
    def inference(self, image)
        pass

class ObjectDetection(ObjecDetectionInterface):
    def __init__(self):

        if torch.backend.mps.is_available():
            log.warning("MPS backend is available, but YOLOv8 does not support it. Using CPU instead.")
            self.device =torch.device('mps')
        elif torch.cuda.is_available():
            log.info("Using CUDA for object detection.")
            self.device = torch.device('cuda')
        else:
            log.info("Using CPU for object detection.")
            self.device = torch.device('cpu')
        
        

        # model
        object_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/yolo11s.pt')
        self.object_model: YOLO = YOLO(object_model_path).to(self.device)


    def inference(self, image: np.ndarray) -> tuple(list[Results], dict[int, str]):
        return self.object_model(image, conf=0.25, iou=0.45, verbose=False, persist=True, imgsz=640, string=True).results, self.object_model.names
