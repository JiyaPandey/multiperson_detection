"""
UI Dashboard Module
Real-time visualization dashboard
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class Dashboard:
    """Real-time visualization dashboard"""
    
    def __init__(self, window_name: str = "Multi-Person Detection Dashboard"):
        """
        Initialize dashboard
        
        Args:
            window_name: Name of the display window
        """
        self.window_name = window_name
        self.stats_panel_width = 300
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    
    def render(self, frame: np.ndarray, tracks: List, 
               statistics: Dict, show_ids: bool = True,
               show_trajectories: bool = True,
               heatmap_overlay: Optional[np.ndarray] = None,
               trajectory_overlay: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render dashboard with all visualizations
        
        Args:
            frame: Input frame
            tracks: List of Track objects
            statistics: Statistics dictionary
            show_ids: Whether to show ID labels
            show_trajectories: Whether to show trajectory trails
            heatmap_overlay: Optional heatmap overlay
            trajectory_overlay: Optional trajectory overlay
            
        Returns:
            Rendered dashboard frame
        """
        # Start with base frame
        output = frame.copy()
        
        # Apply heatmap overlay if available
        if heatmap_overlay is not None:
            output = heatmap_overlay
        
        # Apply trajectory overlay if available
        if trajectory_overlay is not None and show_trajectories:
            output = trajectory_overlay
        
        # Draw bounding boxes and IDs
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            
            # Get color based on track ID
            color = self._get_color_for_id(track.track_id)
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID label
            if show_ids:
                label = f"ID: {track.track_id}"
                
                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
                
                # Draw label text
                cv2.putText(output, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw trajectory trail on the frame
            if show_trajectories and len(track.history) > 1:
                points = list(track.history)
                for i in range(len(points) - 1):
                    cv2.line(output, points[i], points[i + 1], color, 2)
        
        # Add statistics panel
        output = self._add_stats_panel(output, statistics, len(tracks))
        
        return output
    
    def _add_stats_panel(self, frame: np.ndarray, statistics: Dict,
                         active_count: int) -> np.ndarray:
        """
        Add statistics panel to the side
        
        Args:
            frame: Input frame
            statistics: Statistics dictionary
            active_count: Number of active tracks
            
        Returns:
            Frame with stats panel
        """
        height, width = frame.shape[:2]
        
        # Create panel
        panel = np.zeros((height, self.stats_panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background
        
        # Add title
        cv2.putText(panel, "Statistics", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add separator line
        cv2.line(panel, (10, 40), (self.stats_panel_width - 10, 40), (255, 255, 255), 1)
        
        # Add statistics
        y_offset = 70
        line_height = 30
        
        stats_to_show = [
            ("Active IDs:", str(active_count)),
            ("Total IDs:", str(statistics.get('unique_ids', 0))),
            ("Frame:", str(statistics.get('frame_number', 0))),
            ("FPS:", f"{statistics.get('fps', 0):.1f}"),
        ]
        
        for label, value in stats_to_show:
            cv2.putText(panel, label, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(panel, value, (150, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # Combine frame and panel
        combined = np.hstack([frame, panel])
        
        return combined
    
    def show(self, frame: np.ndarray, wait_key: int = 1) -> int:
        """
        Display frame in window
        
        Args:
            frame: Frame to display
            wait_key: Wait time in milliseconds
            
        Returns:
            Key pressed (or -1)
        """
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(wait_key)
    
    def close(self):
        """Close dashboard window"""
        cv2.destroyWindow(self.window_name)
    
    @staticmethod
    def _get_color_for_id(track_id: int) -> Tuple[int, int, int]:
        """
        Get consistent color for track ID
        
        Args:
            track_id: Track identifier
            
        Returns:
            BGR color tuple
        """
        # Generate color based on ID
        np.random.seed(track_id)
        color = tuple(np.random.randint(50, 255, 3).tolist())
        return color
    
    def draw_grid_lines(self, frame: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
        """
        Draw grid lines on frame
        
        Args:
            frame: Input frame
            grid_size: (rows, cols)
            
        Returns:
            Frame with grid lines
        """
        output = frame.copy()
        height, width = frame.shape[:2]
        rows, cols = grid_size
        
        # Draw vertical lines
        for col in range(1, cols):
            x = col * width // cols
            cv2.line(output, (x, 0), (x, height), (0, 255, 0), 2)
        
        # Draw horizontal lines
        for row in range(1, rows):
            y = row * height // rows
            cv2.line(output, (0, y), (width, y), (0, 255, 0), 2)
        
        return output
