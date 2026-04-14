"""
Grid Splitter Module
Splits video frames into grid cells for multi-camera simulation
"""

import numpy as np
from typing import List, Tuple


class GridSplitter:
    """Splits frames into grid-based camera regions"""
    
    def __init__(self, grid_size: Tuple[int, int]):
        """
        Initialize grid splitter
        
        Args:
            grid_size: Tuple of (rows, cols) for grid layout
        """
        self.rows, self.cols = grid_size
        self.num_cameras = self.rows * self.cols
        
        print(f"Grid splitter initialized: {self.rows}x{self.cols} = {self.num_cameras} cameras")
    
    def split_frame(self, frame: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """
        Split frame into grid cells
        
        Args:
            frame: Input frame to split
            
        Returns:
            List of (camera_id, sub_frame) tuples
        """
        height, width = frame.shape[:2]
        
        # Calculate cell dimensions
        cell_height = height // self.rows
        cell_width = width // self.cols
        
        cells = []
        camera_id = 0
        
        for row in range(self.rows):
            for col in range(self.cols):
                # Calculate cell boundaries
                y_start = row * cell_height
                y_end = (row + 1) * cell_height if row < self.rows - 1 else height
                
                x_start = col * cell_width
                x_end = (col + 1) * cell_width if col < self.cols - 1 else width
                
                # Extract cell
                cell = frame[y_start:y_end, x_start:x_end].copy()
                
                cells.append((camera_id, cell))
                camera_id += 1
        
        return cells
    
    def get_camera_position(self, camera_id: int) -> Tuple[int, int]:
        """
        Get grid position (row, col) for camera ID
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Tuple of (row, col)
        """
        row = camera_id // self.cols
        col = camera_id % self.cols
        return (row, col)
    
    def reconstruct_frame(self, cells: List[Tuple[int, np.ndarray]], 
                          original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstruct full frame from grid cells
        
        Args:
            cells: List of (camera_id, annotated_cell) tuples
            original_shape: Original frame (height, width)
            
        Returns:
            Reconstructed frame
        """
        height, width = original_shape
        
        # Sort cells by camera_id
        cells = sorted(cells, key=lambda x: x[0])
        
        # Get cell dimensions from first cell
        if not cells:
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        cell_height, cell_width = cells[0][1].shape[:2]
        
        # Create output frame
        output = np.zeros((height, width, 3), dtype=np.uint8)
        
        for camera_id, cell in cells:
            row, col = self.get_camera_position(camera_id)
            
            y_start = row * cell_height
            y_end = min(y_start + cell.shape[0], height)
            
            x_start = col * cell_width
            x_end = min(x_start + cell.shape[1], width)
            
            output[y_start:y_end, x_start:x_end] = cell[:y_end-y_start, :x_end-x_start]
        
        return output
