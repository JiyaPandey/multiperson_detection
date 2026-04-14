"""
Video Input Module
Handles video stream reading and frame extraction
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class VideoInput:
    """Manages video input reading and preprocessing"""
    
    def __init__(self, video_path: str, target_resolution: Optional[int] = None):
        """
        Initialize video input handler
        
        Args:
            video_path: Path to input video file
            target_resolution: Target height resolution (e.g., 1080, 720)
        """
        self.video_path = Path(video_path)
        self.target_resolution = target_resolution
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video loaded: {self.width}x{self.height} @ {self.fps}fps, {self.frame_count} frames")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from video
        
        Returns:
            Tuple of (success, frame)
        """
        ret, frame = self.cap.read()
        
        if not ret:
            return False, None
        
        # Resize if target resolution specified
        if self.target_resolution and self.height != self.target_resolution:
            scale = self.target_resolution / self.height
            new_width = int(self.width * scale)
            frame = cv2.resize(frame, (new_width, self.target_resolution))
        
        return True, frame
    
    def get_current_frame_number(self) -> int:
        """Get current frame number"""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def reset(self):
        """Reset video to beginning"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def release(self):
        """Release video capture resources"""
        if self.cap:
            self.cap.release()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
