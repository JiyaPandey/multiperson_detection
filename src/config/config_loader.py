"""
Configuration Loader Module
Handles YAML configuration parsing and validation
"""

import yaml
from typing import Dict, Any, Tuple
from pathlib import Path


class ConfigLoader:
    """Loads and validates YAML configuration for the system"""
    
    def __init__(self, config_path: str):
        """
        Initialize the config loader
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration from file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self):
        """Validate required configuration fields"""
        required_sections = ['input', 'processing', 'tracking', 'output']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate input section
        if self.config['input']['type'] not in ['single', 'grid']:
            raise ValueError("input.type must be 'single' or 'grid'")
        
        if self.config['input']['type'] == 'grid' and 'grid_size' not in self.config['input']:
            raise ValueError("grid_size required when input.type is 'grid'")
    
    def get(self, key: str, default=None) -> Any:
        """
        Get configuration value by key (supports dot notation)
        
        Args:
            key: Configuration key (e.g., 'input.type')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def is_grid_mode(self) -> bool:
        """Check if grid mode is enabled"""
        return self.get('input.type') == 'grid'
    
    def get_grid_size(self) -> Tuple[int, int]:
        """Get grid dimensions (rows, cols)"""
        if self.is_grid_mode():
            grid_size = self.get('input.grid_size', [2, 2])
            return tuple(grid_size)
        return (1, 1)
    
    def get_video_path(self) -> str:
        """Get input video path"""
        return self.get('input.video_path')
    
    def use_gpu(self) -> bool:
        """Check if GPU should be used"""
        return self.get('processing.use_gpu', False)
    
    def get_tracking_algorithm(self) -> str:
        """Get tracking algorithm name"""
        return self.get('tracking.algorithm', 'deepsort')
    
    def is_reid_enabled(self) -> bool:
        """Check if ReID is enabled"""
        return self.get('reid.enabled', False)
    
    def show_ui(self) -> bool:
        """Check if UI should be displayed"""
        return self.get('output.show_ui', True)
    
    def save_video(self) -> bool:
        """Check if output video should be saved"""
        return self.get('output.save_video', True)
    
    def generate_heatmap(self) -> bool:
        """Check if heatmap should be generated"""
        return self.get('output.heatmap', True)
    
    def generate_trajectories(self) -> bool:
        """Check if trajectories should be generated"""
        return self.get('output.trajectories', True)
