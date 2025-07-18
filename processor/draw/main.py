# processor/draw/main.py
from abc import ABC, abstractmethod
from typing import List, Dict
from collections import defaultdict
import numpy as np
import cv2

from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator, colors


class TrackDrawer(TrackDrawerInterface):
    def __init__(self):
        self.track_history = defaultdict(list)
        self.thickness: int = 2

    def draw(self, image: np.ndarray, track_ids: List[int], boxes: np.ndarray) -> np.ndarray:
        for track_id, box in zip(track_ids, boxes):
            track_line = self.track_history[track_id]
            centroid = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
            track_line.append(centroid)

            # Limitar historial para no acumular demasiados puntos
            if len(track_line) > 50:
                track_line.pop(0)

            for i in range(1, len(track_line)):
                cv2.line(
                    image,
                    tuple(map(int, track_line[i - 1])),
                    tuple(map(int, track_line[i])),
                    colors(track_id, True),
                    self.thickness
                )
        return image


class DrawingInterface(ABC):
    @abstractmethod
    def draw(self, image: np.ndarray, trash_track: Results, trash_classes: Dict[int, str]) -> np.ndarray:
        raise NotImplementedError


class Drawing(DrawingInterface):
    def __init__(self):
        self.mask_drawer: MaskDrawerInterface = MaskDrawer()
        self.bbox_drawer: BoundingBoxDrawerInterface = BoundingBoxDrawer()
        self.track_drawer: TrackDrawerInterface = TrackDrawer()

    def draw(self, image: np.ndarray, object_track: Results, object_classes: Dict[int, str]) -> np.ndarray:
        # Obtener cajas, clases y IDs
        boxes = object_track.boxes.xyxy.cpu().numpy()
        classes = object_track.boxes.cls.cpu().tolist()
        tracks_ids = object_track.boxes.id.int().cpu().tolist() if object_track.boxes.id is not None else []

        # Dibujar máscaras si existen
        if hasattr(object_track, "masks") and object_track.masks is not None:
            masks = object_track.masks.xy
            image = self.mask_drawer.draw(image, masks, classes)

        # Dibujar cajas
        image = self.bbox_drawer.draw(image, boxes, object_classes, classes)

        # Dibujar trayectorias si hay IDs
        if tracks_ids:
            image = self.track_drawer.draw(image, tracks_ids, boxes)

        return image
