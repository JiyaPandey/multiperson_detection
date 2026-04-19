"""
Visualization Utilities
Color generation and drawing functions for tracking
"""

import cv2
import random
import numpy as np


# Global color storage for consistent colors
_id_colors = {}


def get_color(gid, seed=True):
    """
    Generate consistent random color for a global ID
    
    Args:
        gid: Global ID
        seed: Whether to use ID as seed for consistency
        
    Returns:
        BGR color tuple
    """
    if gid in _id_colors:
        return _id_colors[gid]
    
    if seed:
        random.seed(int(gid))
    
    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    _id_colors[gid] = color
    
    return color


def draw_bbox(frame, box, gid, color=None, thickness=2):
    """
    Draw bounding box with ID label
    
    Args:
        frame: Input frame
        box: Bounding box (x1, y1, x2, y2)
        gid: Global ID
        color: Optional color override
        thickness: Line thickness
        
    Returns:
        Frame with drawn bbox
    """
    x1, y1, x2, y2 = map(int, box)
    
    if color is None:
        color = get_color(gid)
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label
    label = f"ID {gid}"
    draw_label(frame, label, (x1, y1 - 10), color)
    
    return frame


def draw_label(frame, text, position, color, font_scale=0.6, thickness=2):
    """
    Draw text label with background
    
    Args:
        frame: Input frame
        text: Label text
        position: Position (x, y)
        color: Label background color
        font_scale: Font scale
        thickness: Text thickness
        
    Returns:
        Frame with drawn label
    """
    x, y = position
    
    # Get text size
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(frame, (x, y - th - 6), (x + tw, y), color, -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return frame


def reset_colors():
    """Reset color cache"""
    global _id_colors
    _id_colors.clear()
