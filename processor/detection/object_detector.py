# processor/detection/object_detector.py
import os
import logging as log
import torch
import numpy as np
from abc import ABC, abstractmethod
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List, Tuple, Dict

class ObjecDetectionInterface(ABC):
    @abstractmethod
    def inference(self, image: np.ndarray) -> Tuple[List[Results], Dict[int, str]]:
        pass

class ObjectDetection(ObjecDetectionInterface):
    def __init__(self):
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            log.info(f"✅ GPU disponible: {device_name}")
            self.device = torch.device('cuda')
        else:
            log.warning("⚠️ GPU no disponible, usando CPU")
            self.device = torch.device('cpu')

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/yolo11s.pt')
        log.info(f"📦 Cargando modelo YOLO11 desde: {model_path}")
        self.object_model: YOLO = YOLO(model_path).to(self.device)
        log.info(f"🚀 Modelo cargado en dispositivo: {self.device}")

    def inference(self, image: np.ndarray) -> Tuple[List[Results], Dict[int, str]]:
        results = self.object_model(image, conf=0.25, iou=0.45, verbose=True, imgsz=640, device=str(self.device))
        return results, self.object_model.names
