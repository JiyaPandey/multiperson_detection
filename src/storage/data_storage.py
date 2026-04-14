"""
Data Storage Module
Stores tracking and position data for analytics
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import json


class DataPoint:
    """Represents a single tracking data point"""
    
    def __init__(self, global_id: int, camera_id: int, timestamp: float,
                 position: Tuple[int, int], bbox: Tuple[int, int, int, int],
                 frame_number: int):
        """
        Initialize data point
        
        Args:
            global_id: Global person ID
            camera_id: Camera identifier
            timestamp: Timestamp in seconds
            position: Center position (x, y)
            bbox: Bounding box (x1, y1, x2, y2)
            frame_number: Frame number
        """
        self.global_id = global_id
        self.camera_id = camera_id
        self.timestamp = timestamp
        self.position = position
        self.bbox = bbox
        self.frame_number = frame_number
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'global_id': self.global_id,
            'camera_id': self.camera_id,
            'timestamp': self.timestamp,
            'position': self.position,
            'bbox': self.bbox,
            'frame_number': self.frame_number
        }


class DataStorage:
    """In-memory storage for tracking data"""
    
    def __init__(self):
        """Initialize data storage"""
        # Store all data points
        self.data_points: List[DataPoint] = []
        
        # Index by global_id for quick lookup
        self.by_global_id: Dict[int, List[DataPoint]] = defaultdict(list)
        
        # Index by camera_id
        self.by_camera_id: Dict[int, List[DataPoint]] = defaultdict(list)
        
        # Store start time
        self.start_time = datetime.now()
    
    def add(self, global_id: int, camera_id: int, position: Tuple[int, int],
            bbox: Tuple[int, int, int, int], frame_number: int):
        """
        Add new data point
        
        Args:
            global_id: Global person ID
            camera_id: Camera identifier
            position: Center position (x, y)
            bbox: Bounding box (x1, y1, x2, y2)
            frame_number: Frame number
        """
        # Calculate timestamp
        timestamp = (datetime.now() - self.start_time).total_seconds()
        
        # Create data point
        point = DataPoint(global_id, camera_id, timestamp, position, bbox, frame_number)
        
        # Store
        self.data_points.append(point)
        self.by_global_id[global_id].append(point)
        self.by_camera_id[camera_id].append(point)
    
    def get_trajectory(self, global_id: int) -> List[Tuple[int, int]]:
        """
        Get position trajectory for a person
        
        Args:
            global_id: Global person ID
            
        Returns:
            List of (x, y) positions over time
        """
        points = self.by_global_id.get(global_id, [])
        return [p.position for p in points]
    
    def get_all_positions(self, camera_id: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Get all positions, optionally filtered by camera
        
        Args:
            camera_id: Optional camera filter
            
        Returns:
            List of (x, y) positions
        """
        if camera_id is not None:
            points = self.by_camera_id.get(camera_id, [])
        else:
            points = self.data_points
        
        return [p.position for p in points]
    
    def get_statistics(self) -> Dict:
        """Get storage statistics"""
        return {
            'total_points': len(self.data_points),
            'unique_ids': len(self.by_global_id),
            'cameras': len(self.by_camera_id),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds()
        }
    
    def export_to_json(self, filepath: str):
        """
        Export data to JSON file
        
        Args:
            filepath: Output JSON file path
        """
        data = {
            'metadata': self.get_statistics(),
            'data_points': [p.to_dict() for p in self.data_points]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Data exported to: {filepath}")
    
    def clear(self):
        """Clear all stored data"""
        self.data_points.clear()
        self.by_global_id.clear()
        self.by_camera_id.clear()
        self.start_time = datetime.now()
