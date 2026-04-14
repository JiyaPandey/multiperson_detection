"""
Trajectory Builder Module
Tracks and visualizes movement paths
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from collections import deque, defaultdict


class TrajectoryBuilder:
    """Builds and visualizes movement trajectories"""
    
    def __init__(self, max_length: int = 100):
        """
        Initialize trajectory builder
        
        Args:
            max_length: Maximum trajectory length to store
        """
        self.max_length = max_length
        
        # Store trajectories: global_id -> deque of (x, y) positions
        self.trajectories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=max_length)
        )
        
        # Color mapping for trajectories
        self.colors = {}
        self._color_palette = self._generate_color_palette(50)
    
    def add_position(self, global_id: int, position: Tuple[int, int]):
        """
        Add position to trajectory
        
        Args:
            global_id: Global person ID
            position: (x, y) position
        """
        self.trajectories[global_id].append(position)
        
        # Assign color if new ID
        if global_id not in self.colors:
            color_idx = global_id % len(self._color_palette)
            self.colors[global_id] = self._color_palette[color_idx]
    
    def get_trajectory(self, global_id: int) -> List[Tuple[int, int]]:
        """
        Get trajectory for a person
        
        Args:
            global_id: Global person ID
            
        Returns:
            List of (x, y) positions
        """
        return list(self.trajectories.get(global_id, []))
    
    def draw_trajectories(self, frame: np.ndarray, 
                          active_ids: List[int] = None,
                          thickness: int = 2) -> np.ndarray:
        """
        Draw trajectories on frame
        
        Args:
            frame: Frame to draw on
            active_ids: List of IDs to draw (None = all)
            thickness: Line thickness
            
        Returns:
            Frame with trajectories drawn
        """
        output = frame.copy()
        
        # Determine which IDs to draw
        ids_to_draw = active_ids if active_ids else self.trajectories.keys()
        
        for global_id in ids_to_draw:
            trajectory = self.trajectories.get(global_id)
            
            if not trajectory or len(trajectory) < 2:
                continue
            
            # Get color
            color = self.colors.get(global_id, (255, 255, 255))
            
            # Draw trajectory lines
            points = list(trajectory)
            for i in range(len(points) - 1):
                # Fade older points
                alpha = (i + 1) / len(points)
                current_thickness = max(1, int(thickness * alpha))
                
                cv2.line(output, points[i], points[i + 1], color, current_thickness)
            
            # Draw circle at current position
            if points:
                cv2.circle(output, points[-1], 5, color, -1)
        
        return output
    
    def draw_single_trajectory(self, frame: np.ndarray, global_id: int,
                               color: Tuple[int, int, int] = None,
                               thickness: int = 2) -> np.ndarray:
        """
        Draw single trajectory on frame
        
        Args:
            frame: Frame to draw on
            global_id: Person ID
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            Frame with trajectory drawn
        """
        output = frame.copy()
        trajectory = self.trajectories.get(global_id)
        
        if not trajectory or len(trajectory) < 2:
            return output
        
        # Use default color if not specified
        if color is None:
            color = self.colors.get(global_id, (255, 255, 255))
        
        # Draw trajectory
        points = list(trajectory)
        for i in range(len(points) - 1):
            cv2.line(output, points[i], points[i + 1], color, thickness)
        
        return output
    
    def clear_trajectory(self, global_id: int):
        """Clear trajectory for specific ID"""
        if global_id in self.trajectories:
            self.trajectories[global_id].clear()
    
    def reset(self):
        """Clear all trajectories"""
        self.trajectories.clear()
        self.colors.clear()
    
    @staticmethod
    def _generate_color_palette(n: int) -> List[Tuple[int, int, int]]:
        """
        Generate distinct colors
        
        Args:
            n: Number of colors to generate
            
        Returns:
            List of BGR tuples
        """
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, bgr)))
        return colors
