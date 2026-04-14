"""
Person Detection Module
Uses YOLOv8 for person detection in video frames
"""

import numpy as np
from typing import List, Tuple
from ultralytics import YOLO


class Detection:
    """Represents a single person detection"""
    
    def __init__(self, bbox: Tuple[float, float, float, float], 
                 confidence: float, class_id: int = 0):
        """
        Initialize detection
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence score
            class_id: Class ID (0 for person)
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
    
    def get_xyxy(self) -> Tuple[int, int, int, int]:
        """Get bounding box in xyxy format"""
        return tuple(map(int, self.bbox))
    
    def get_xywh(self) -> Tuple[int, int, int, int]:
        """Get bounding box in xywh format"""
        x1, y1, x2, y2 = self.bbox
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    
    def get_center(self) -> Tuple[int, int]:
        """Get center point of detection"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))


class PersonDetector:
    """YOLOv8-based person detector"""
    
    def __init__(self, model_name: str = 'yolov8n.pt', 
                 confidence_threshold: float = 0.5,
                 use_gpu: bool = False):
        """
        Initialize person detector
        
        Args:
            model_name: YOLO model name ('yolov8n', 'yolov8s', 'yolov8m', etc.)
            confidence_threshold: Minimum confidence for detections
            use_gpu: Whether to use GPU acceleration
        """
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if use_gpu else 'cpu'
        
        print(f"Loading YOLO model: {model_name} on {self.device}")
        self.model = YOLO(model_name)
        
        # Person class ID in COCO dataset
        self.person_class_id = 0
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect people in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(frame, device=self.device, verbose=False)
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get class ID and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Filter for person class and confidence threshold
                if class_id == self.person_class_id and confidence >= self.confidence_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detection = Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=confidence,
                        class_id=class_id
                    )
                    
                    detections.append(detection)
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect people in multiple frames
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection lists, one per frame
        """
        return [self.detect(frame) for frame in frames]
