"""
Video Input Module
Unified frame loading interface for all sources
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple, List, Iterator
from pathlib import Path


class FrameLoader:
    """Base class for frame loaders"""
    
    def __iter__(self):
        return self
    
    def __next__(self):
        raise NotImplementedError
    
    def get_properties(self):
        raise NotImplementedError


class VideoLoader(FrameLoader):
    """Load frames from video file"""
    
    def __init__(self, video_path: str, target_resolution: Optional[int] = None):
        """
        Args:
            video_path: Path to video file
            target_resolution: Optional target height for resizing
        """
        self.video_path = Path(video_path)
        self.target_resolution = target_resolution
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        # Properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video loaded: {self.width}x{self.height} @ {self.fps}fps")
    
    def __next__(self):
        ret, frame = self.cap.read()
        
        if not ret:
            raise StopIteration
        
        if self.target_resolution and self.height != self.target_resolution:
            scale = self.target_resolution / self.height
            new_width = int(self.width * scale)
            frame = cv2.resize(frame, (new_width, self.target_resolution))
        
        return frame
    
    def get_properties(self):
        return {
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'frame_count': self.frame_count
        }
    
    def release(self):
        if self.cap:
            self.cap.release()


class MultiCameraLoader(FrameLoader):
    """Load synchronized frames from multiple camera folders"""
    
    def __init__(self, base_path: str, camera_folders: List[str]):
        """
        Args:
            base_path: Base directory containing camera folders
            camera_folders: List of camera folder names
        """
        self.base_path = Path(base_path)
        self.camera_folders = camera_folders
        self.frame_lists = []
        
        # Load all frame paths
        for cam in camera_folders:
            cam_path = self.base_path / cam
            if not cam_path.exists():
                raise FileNotFoundError(f"Camera folder not found: {cam_path}")
            
            frames = sorted(cam_path.glob("*.jpg")) + sorted(cam_path.glob("*.png"))
            frames = [str(f) for f in frames]
            self.frame_lists.append(frames)
        
        # Verify all cameras have same number of frames
        self.num_frames = len(self.frame_lists[0])
        for i, frames in enumerate(self.frame_lists):
            if len(frames) != self.num_frames:
                print(f"Warning: Camera {i} has {len(frames)} frames, expected {self.num_frames}")
        
        self.current_idx = 0
        print(f"Loaded {len(camera_folders)} cameras, {self.num_frames} frames each")
    
    def __next__(self):
        if self.current_idx >= self.num_frames:
            raise StopIteration
        
        # Load frames from all cameras
        frames = []
        for cam_frames in self.frame_lists:
            if self.current_idx < len(cam_frames):
                frame = cv2.imread(cam_frames[self.current_idx])
                frames.append(frame)
            else:
                frames.append(None)
        
        self.current_idx += 1
        return frames, self.current_idx - 1
    
    def get_properties(self):
        return {
            'num_cameras': len(self.camera_folders),
            'num_frames': self.num_frames,
            'current_frame': self.current_idx
        }


def get_loader(source_type: str, **kwargs) -> FrameLoader:
    """
    Factory function to create appropriate frame loader
    
    Args:
        source_type: 'video' or 'multi_camera'
        **kwargs: Arguments passed to loader constructor
        
    Returns:
        FrameLoader instance
    """
    if source_type == 'video':
        return VideoLoader(**kwargs)
    elif source_type == 'multi_camera':
        return MultiCameraLoader(**kwargs)
    else:
        raise ValueError(f"Unknown source type: {source_type}")


# Legacy compatibility
class VideoInput(VideoLoader):
    """Legacy class for backward compatibility"""
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            frame = next(self)
            return True, frame
        except StopIteration:
            return False, None
    
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
