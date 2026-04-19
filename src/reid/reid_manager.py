"""
ReID Manager - Person Re-Identification
Handles feature extraction, matching, and global ID assignment
"""

import cv2
import numpy as np
import torch
from torchreid.reid.utils import FeatureExtractor
from scipy.spatial.distance import cosine
from collections import defaultdict, deque


class ReIDManager:
    """Manages ReID features and global ID matching across cameras"""
    
    def __init__(self, 
                 model_name='osnet_x1_0',
                 device=None,
                 max_features_per_id=20,
                 match_threshold=0.65,
                 spatial_distance_threshold=200,
                 feature_update_interval=3):
        """
        Initialize ReID Manager
        
        Args:
            model_name: ReID model name
            device: Device to use ('cuda' or 'cpu')
            max_features_per_id: Maximum features to store per ID
            match_threshold: Similarity threshold for matching
            spatial_distance_threshold: Spatial consistency threshold
            feature_update_interval: Update features every N frames
        """
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Parameters
        self.max_features_per_id = max_features_per_id
        self.match_threshold = match_threshold
        self.spatial_distance_threshold = spatial_distance_threshold
        self.feature_update_interval = feature_update_interval
        
        # Initialize ReID model
        print(f"Loading ReID model: {model_name} on {device}")
        self.reid_model = FeatureExtractor(model_name=model_name, device=device)
        
        # Storage for global ID features
        self.global_id_features = defaultdict(lambda: deque(maxlen=max_features_per_id))
        
        # Track last known position of each ID per camera
        self.last_positions = defaultdict(lambda: defaultdict(lambda: None))
        
        # Track when each ID was last seen
        self.id_frame_history = defaultdict(int)
        
        # Next available global ID
        self.next_gid = 1
    
    def normalize_feature(self, feat):
        """Normalize feature vector to unit length"""
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat
    
    def extract_feature(self, frame, box):
        """
        Extract ReID feature from detected person crop
        
        Args:
            frame: Input frame
            box: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Normalized feature vector or None
        """
        x1, y1, x2, y2 = map(int, box)
        
        # Add some padding and ensure valid crop
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        
        # Resize to standard ReID input size
        crop = cv2.resize(crop, (128, 256))
        
        # Extract feature
        try:
            feat = self.reid_model([crop])[0].cpu().numpy()
            return self.normalize_feature(feat)
        except:
            return None
    
    def compute_similarity(self, feat1, feat2):
        """
        Compute cosine similarity between two features
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector
            
        Returns:
            Similarity score (higher is more similar)
        """
        return 1 - cosine(feat1, feat2)
    
    def match_to_global_id(self, feature, cam_id, position, frame_idx, used_ids_in_frame=None):
        """
        Match a feature to existing global IDs or create new one
        
        Args:
            feature: Feature vector
            cam_id: Camera ID
            position: Position (x, y)
            frame_idx: Frame index
            used_ids_in_frame: Dict of {cam_id: set of used IDs}
            
        Returns:
            Global ID or None
        """
        if feature is None:
            return None
        
        if used_ids_in_frame is None:
            used_ids_in_frame = {}
        
        candidates = []  # List of (gid, similarity, distance_penalty)
        
        # Compare with all existing global IDs
        for gid, feature_list in self.global_id_features.items():
            if len(feature_list) == 0:
                continue
            
            # Skip if this ID is already used in this frame for this camera
            if gid in used_ids_in_frame.get(cam_id, set()):
                continue
            
            # Compute average similarity with all stored features for this ID
            similarities = [self.compute_similarity(feature, f) for f in feature_list]
            avg_similarity = np.mean(similarities)
            
            # Apply temporal consistency bonus if this ID was recently seen in this camera
            temporal_bonus = 0
            if gid in self.last_positions and cam_id in self.last_positions[gid]:
                last_pos = self.last_positions[gid][cam_id]
                if last_pos is not None:
                    # Calculate distance from last known position
                    dist = np.sqrt((position[0] - last_pos[0])**2 + (position[1] - last_pos[1])**2)
                    # If close to last position, give bonus (person likely hasn't moved much)
                    if dist < self.spatial_distance_threshold:
                        temporal_bonus = 0.05
            
            total_similarity = avg_similarity + temporal_bonus
            candidates.append((gid, total_similarity))
        
        # Find best match
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_gid, best_similarity = candidates[0]
            
            # Match threshold: if similarity is high enough, assign to existing ID
            if best_similarity > self.match_threshold:
                # Add this feature to the existing ID's feature bank
                self.global_id_features[best_gid].append(feature)
                # Update last position
                self.last_positions[best_gid][cam_id] = position
                self.id_frame_history[best_gid] = frame_idx
                return best_gid
        
        # Create new global ID
        new_gid = self.next_gid
        self.global_id_features[new_gid].append(feature)
        self.last_positions[new_gid][cam_id] = position
        self.id_frame_history[new_gid] = frame_idx
        self.next_gid += 1
        return new_gid
    
    def process(self, frame, box, cam_id, position, frame_idx, used_ids_in_frame=None):
        """
        Wrapper: extract feature and match to global ID
        
        Args:
            frame: Input frame
            box: Bounding box (x1, y1, x2, y2)
            cam_id: Camera ID
            position: Position (x, y)
            frame_idx: Frame index
            used_ids_in_frame: Dict of {cam_id: set of used IDs}
            
        Returns:
            Global ID or None
        """
        feature = self.extract_feature(frame, box)
        if feature is None:
            return None
        
        return self.match_to_global_id(feature, cam_id, position, frame_idx, used_ids_in_frame)
    
    def get_active_count(self):
        """Get count of active global IDs"""
        return len(self.global_id_features)
    
    def reset(self):
        """Reset all data"""
        self.global_id_features.clear()
        self.last_positions.clear()
        self.id_frame_history.clear()
        self.next_gid = 1
