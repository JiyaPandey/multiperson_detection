"""
Heatmap Generator Module
Creates activity heatmaps from position data
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class HeatmapGenerator:
    """Generates activity heatmaps from tracking data"""
    
    def __init__(self, frame_size: Tuple[int, int], blur_kernel: int = 51):
        """
        Initialize heatmap generator
        
        Args:
            frame_size: (width, height) of the frame
            blur_kernel: Kernel size for Gaussian blur (must be odd)
        """
        self.width, self.height = frame_size
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        
        # Accumulation matrix
        self.accumulator = np.zeros((self.height, self.width), dtype=np.float32)
    
    def add_positions(self, positions: List[Tuple[int, int]], weight: float = 1.0):
        """
        Add positions to the heatmap accumulator
        
        Args:
            positions: List of (x, y) positions
            weight: Weight for each position
        """
        for x, y in positions:
            # Clip to valid range
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            
            # Add to accumulator
            self.accumulator[y, x] += weight
    
    def add_position(self, position: Tuple[int, int], weight: float = 1.0):
        """Add single position to heatmap"""
        self.add_positions([position], weight)
    
    def generate_heatmap(self, normalize: bool = True) -> np.ndarray:
        """
        Generate heatmap visualization
        
        Args:
            normalize: Whether to normalize values
            
        Returns:
            Heatmap as BGR image
        """
        # Apply Gaussian blur for smooth heatmap
        heatmap = cv2.GaussianBlur(self.accumulator, (self.blur_kernel, self.blur_kernel), 0)
        
        # Normalize
        if normalize and heatmap.max() > 0:
            heatmap = heatmap / heatmap.max() * 255
        
        # Convert to uint8
        heatmap = heatmap.astype(np.uint8)
        
        # Apply colormap (COLORMAP_JET: blue=low, red=high)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap_colored
    
    def get_overlay(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Get heatmap overlay on frame
        
        Args:
            frame: Background frame
            alpha: Transparency (0=fully transparent, 1=fully opaque)
            
        Returns:
            Frame with heatmap overlay
        """
        heatmap = self.generate_heatmap()
        
        # Resize heatmap if needed
        if heatmap.shape[:2] != frame.shape[:2]:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        # Blend with frame
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
    
    def reset(self):
        """Reset accumulator"""
        self.accumulator = np.zeros((self.height, self.width), dtype=np.float32)
    
    def save(self, filepath: str):
        """Save heatmap to file"""
        heatmap = self.generate_heatmap()
        cv2.imwrite(filepath, heatmap)
        print(f"Heatmap saved to: {filepath}")
