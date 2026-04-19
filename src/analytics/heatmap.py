"""
Heatmap Analytics
Activity heatmap generation and visualization
"""

import cv2
import numpy as np


class Heatmap:
    """Manages activity heatmap generation"""
    
    def __init__(self, width, height, decay_factor=0.995, blur_kernel=31, weight=8.0):
        """
        Initialize heatmap
        
        Args:
            width: Heatmap width
            height: Heatmap height
            decay_factor: Decay factor for temporal smoothing
            blur_kernel: Gaussian blur kernel size (odd number)
            weight: Weight for each detection
        """
        self.width = width
        self.height = height
        self.decay_factor = decay_factor
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.weight = weight
        
        # Heatmap accumulator
        self.heatmap = np.zeros((height, width), dtype=np.float32)
    
    def update(self, detections):
        """
        Update heatmap with new detections
        
        Args:
            detections: List of (x, y) positions
        """
        # Add detections
        for x, y in detections:
            map_x = int(x)
            map_y = int(y)
            
            if 0 <= map_x < self.width and 0 <= map_y < self.height:
                self.heatmap[map_y, map_x] += self.weight
        
        # Apply decay
        self.heatmap *= self.decay_factor
    
    def render(self, colormap=cv2.COLORMAP_TURBO, background_color=(255, 255, 255), threshold=30):
        """
        Render heatmap as colored image
        
        Args:
            colormap: OpenCV colormap
            background_color: Background color for low-intensity areas
            threshold: Intensity threshold for background
            
        Returns:
            Colored heatmap image
        """
        if np.max(self.heatmap) > 0:
            # Apply Gaussian blur
            blur = cv2.GaussianBlur(self.heatmap, (self.blur_kernel, self.blur_kernel), 0)
            
            # Normalize
            norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Apply colormap
            color = cv2.applyColorMap(norm, colormap)
            
            # Remove weak noise
            color[norm < threshold] = background_color
            
            return color
        else:
            # Return blank image
            return np.ones((self.height, self.width, 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)
    
    def overlay(self, frame, alpha=0.5):
        """
        Overlay heatmap on frame
        
        Args:
            frame: Background frame
            alpha: Overlay transparency (0=transparent, 1=opaque)
            
        Returns:
            Frame with heatmap overlay
        """
        heatmap_img = self.render()
        
        # Resize if needed
        if heatmap_img.shape[:2] != frame.shape[:2]:
            heatmap_img = cv2.resize(heatmap_img, (frame.shape[1], frame.shape[0]))
        
        # Blend
        return cv2.addWeighted(frame, 1 - alpha, heatmap_img, alpha, 0)
    
    def reset(self):
        """Reset heatmap"""
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)
