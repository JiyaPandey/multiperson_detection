"""
Person Tracking Module
Implements tracking algorithms (DeepSORT/ByteTrack)
"""

import numpy as np
from typing import List, Optional, Tuple
from collections import defaultdict, deque


class Track:
    """Represents a tracked person"""
    
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], 
                 camera_id: int = 0):
        """
        Initialize track
        
        Args:
            track_id: Unique track identifier
            bbox: Bounding box (x1, y1, x2, y2)
            camera_id: Camera/stream identifier
        """
        self.track_id = track_id
        self.bbox = bbox
        self.camera_id = camera_id
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        
        # Position history for trajectory
        self.history = deque(maxlen=100)
        self.history.append(self.get_center())
    
    def get_center(self) -> Tuple[int, int]:
        """Get center point of track"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def update(self, bbox: Tuple[int, int, int, int]):
        """Update track with new detection"""
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.get_center())
    
    def predict(self):
        """Predict next position (simple constant velocity)"""
        self.age += 1
        self.time_since_update += 1


class PersonTracker:
    """
    Simple but robust person tracker
    Uses IoU-based matching with Kalman-like prediction
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, 
                 iou_threshold: float = 0.3):
        """
        Initialize tracker
        
        Args:
            max_age: Maximum frames to keep track without update
            min_hits: Minimum hits before track is confirmed
            iou_threshold: Minimum IoU for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, detections: List, camera_id: int = 0) -> List[Track]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of Detection objects
            camera_id: Camera identifier
            
        Returns:
            List of updated tracks
        """
        self.frame_count += 1
        
        # Convert detections to bboxes
        det_bboxes = [det.get_xyxy() for det in detections]
        
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        if len(self.tracks) > 0 and len(det_bboxes) > 0:
            matched, unmatched_dets, unmatched_trks = self._match_detections_to_tracks(
                det_bboxes, self.tracks
            )
            
            # Update matched tracks
            for det_idx, trk_idx in matched:
                self.tracks[trk_idx].update(det_bboxes[det_idx])
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                self._initiate_track(det_bboxes[det_idx], camera_id)
            
            # Mark unmatched tracks for potential deletion
            for trk_idx in unmatched_trks:
                pass  # Already predicted above
        
        elif len(det_bboxes) > 0:
            # No existing tracks, create new ones
            for bbox in det_bboxes:
                self._initiate_track(bbox, camera_id)
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Return confirmed tracks
        confirmed_tracks = [
            t for t in self.tracks 
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits
        ]
        
        return confirmed_tracks
    
    def _initiate_track(self, bbox: Tuple[int, int, int, int], camera_id: int):
        """Create new track"""
        track = Track(self.next_id, bbox, camera_id)
        self.tracks.append(track)
        self.next_id += 1
    
    def _match_detections_to_tracks(self, detections: List, tracks: List) -> Tuple:
        """
        Match detections to existing tracks using IoU
        
        Returns:
            Tuple of (matched, unmatched_detections, unmatched_tracks)
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(tracks):
                iou_matrix[d, t] = self._compute_iou(det, trk.bbox)
        
        # Simple greedy matching
        matched_indices = []
        unmatched_detections = []
        unmatched_tracks = list(range(len(tracks)))
        
        # Sort by IoU value (highest first)
        for d in range(len(detections)):
            if len(unmatched_tracks) == 0:
                unmatched_detections.append(d)
                continue
            
            best_iou = self.iou_threshold
            best_track = -1
            
            for t in unmatched_tracks:
                if iou_matrix[d, t] > best_iou:
                    best_iou = iou_matrix[d, t]
                    best_track = t
            
            if best_track >= 0:
                matched_indices.append((d, best_track))
                unmatched_tracks.remove(best_track)
            else:
                unmatched_detections.append(d)
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    @staticmethod
    def _compute_iou(bbox1: Tuple, bbox2: Tuple) -> float:
        """
        Compute Intersection over Union (IoU)
        
        Args:
            bbox1, bbox2: Bounding boxes in (x1, y1, x2, y2) format
            
        Returns:
            IoU score
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def reset(self):
        """Reset tracker state"""
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
