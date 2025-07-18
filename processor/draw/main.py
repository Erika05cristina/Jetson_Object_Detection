# processor/draw/main.py
from abc import ABC, abstractmethod
from typing import List, Dict
from collections import defaultdict
import numpy as np
import cv2

from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator, colors


# Interfaces
class MaskDrawerInterface(ABC):
    @abstractmethod
    def draw(self, image: np.ndarray, masks: List[np.ndarray], classes: List[int]) -> np.ndarray:
        raise NotImplementedError


class BoundingBoxDrawerInterface(ABC):
    @abstractmethod
    def draw(self, image: np.ndarray, boxes: np.ndarray, object_classes: Dict[int, str], classes: List[int]) -> np.ndarray:
        raise NotImplementedError


class TrackDrawerInterface(ABC):
    @abstractmethod
    def draw(self, image: np.ndarray, track_ids: List[int], boxes: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# Implementaciones concretas
class MaskDrawer(MaskDrawerInterface):
    def __init__(self):
        # Puedes definir un mapa de colores si quieres
        self.color_map = {
            0: (255, 0, 0),
            1: (0, 255, 0),
            2: (0, 0, 255),
        }

    def draw(self, image: np.ndarray, masks: List[np.ndarray], classes: List[int]) -> np.ndarray:
        overlay = image.copy()
        alpha = 0.5
        for mask, cls in zip(masks, classes):
            color = self.color_map.get(cls, (255, 255, 255))
            mask_polygon = np.array(mask, np.int32)
            cv2.fillPoly(overlay, [mask_polygon], color)
            cv2.polylines(image, [mask_polygon], isClosed=True, color=color, thickness=2)
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

class BoundingBoxDrawer(BoundingBoxDrawerInterface):
    def __init__(self):
        self.thickness = 2

    def draw(self, image: np.ndarray, boxes: np.ndarray, object_classes: Dict[int, str], classes: List[int]) -> np.ndarray:
        annotator = Annotator(image)
        for box, cls in zip(boxes, classes):
            label = object_classes.get(cls, str(cls))
            annotator.box_label(box, label, color=colors(cls, True))  # thickness eliminado
        return annotator.result()


class TrackDrawer(TrackDrawerInterface):
    def __init__(self):
        self.track_history = defaultdict(list)
        self.thickness = 2

    def draw(self, image: np.ndarray, track_ids: List[int], boxes: np.ndarray) -> np.ndarray:
        for track_id, box in zip(track_ids, boxes):
            track_line = self.track_history[track_id]
            centroid = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
            track_line.append(centroid)

            if len(track_line) > 50:
                track_line.pop(0)

            for i in range(1, len(track_line)):
                cv2.line(
                    image,
                    track_line[i - 1],
                    track_line[i],
                    colors(track_id, True),
                    self.thickness
                )
        return image


class DrawingInterface(ABC):
    @abstractmethod
    def draw(self, image: np.ndarray, object_track: Results, object_classes: Dict[int, str]) -> np.ndarray:
        raise NotImplementedError


class Drawing(DrawingInterface):
    def __init__(self):
        self.mask_drawer: MaskDrawerInterface = MaskDrawer()
        self.bbox_drawer: BoundingBoxDrawerInterface = BoundingBoxDrawer()
        self.track_drawer: TrackDrawerInterface = TrackDrawer()

    def draw(self, image: np.ndarray, object_track: Results, object_classes: Dict[int, str]) -> np.ndarray:
        boxes = object_track.boxes.xyxy.cpu().numpy()
        classes = object_track.boxes.cls.cpu().tolist()
        tracks_ids = object_track.boxes.id.int().cpu().tolist() if object_track.boxes.id is not None else []

        if hasattr(object_track, "masks") and object_track.masks is not None:
            masks = object_track.masks.xy
            image = self.mask_drawer.draw(image, masks, classes)

        image = self.bbox_drawer.draw(image, boxes, object_classes, classes)

        if tracks_ids:
            image = self.track_drawer.draw(image, tracks_ids, boxes)

        return image
