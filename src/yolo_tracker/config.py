"""
Configuration management for YOLO tracker.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for YOLO tracker."""
    
    DEFAULT_CONFIG = {
        # Model settings
        'model': {
            'yolo_model': 'yolov8n.pt',
            'tracker_type': 'botsort',
            'device': 'auto',  # 'auto', 'cpu', 'cuda'
            'confidence_threshold': 0.5,
            'iou_threshold': 0.7
        },
        
        # Tracker settings
        'tracker': {
            'botsort': {
                'track_high_thresh': 0.5,
                'track_low_thresh': 0.1,
                'new_track_thresh': 0.6,
                'track_buffer': 30,
                'match_thresh': 0.8,
                'frame_rate': 30
            }
        },
        
        # Dataset settings
        'dataset': {
            'sequences_dir': 'sequences',
            'detections_dir': 'detections',
            'results_dir': 'tracking_results',
            'ground_truth_dir': 'gt',
            'image_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
            'default_frame_rate': 30
        },
        
        # Evaluation settings
        'evaluation': {
            'metrics': ['HOTA', 'CLEAR', 'Identity'],
            'benchmark': 'MOT17',
            'split': 'train',
            'use_parallel': False,
            'num_parallel_cores': 1,
            'output_detailed': True
        },
        
        # Video processing settings
        'video': {
            'output_codec': 'mp4v',
            'default_fps': 30,
            'show_progress': True,
            'save_frames': False
        },
        
        # Paths
        'paths': {
            'data_root': './data',
            'output_root': './output',
            'models_dir': './models',
            'temp_dir': './temp'
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path:
            self.load_config(config_path)
            
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Resolve device setting
        self._resolve_device()
        
    def load_config(self, config_path: str):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                file_config = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
        # Merge with default config
        self._deep_update(self.config, file_config)
        
    def save_config(self, config_path: str):
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.yolo_model')
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
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.yolo_model')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with a dictionary.
        
        Args:
            updates: Dictionary of updates
        """
        self._deep_update(self.config, updates)
        
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update base_dict with update_dict."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
                
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Model device override
        if 'YOLO_DEVICE' in os.environ:
            self.set('model.device', os.environ['YOLO_DEVICE'])
            
        # Data root override
        if 'YOLO_DATA_ROOT' in os.environ:
            self.set('paths.data_root', os.environ['YOLO_DATA_ROOT'])
            
        # Output root override
        if 'YOLO_OUTPUT_ROOT' in os.environ:
            self.set('paths.output_root', os.environ['YOLO_OUTPUT_ROOT'])
            
    def _resolve_device(self):
        """Resolve device setting."""
        device = self.get('model.device')
        
        if device == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    self.set('model.device', 'cuda')
                else:
                    self.set('model.device', 'cpu')
            except ImportError:
                self.set('model.device', 'cpu')
                
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get('model', {})
        
    def get_tracker_config(self) -> Dict[str, Any]:
        """Get tracker configuration."""
        tracker_type = self.get('model.tracker_type', 'botsort')
        return self.get(f'tracker.{tracker_type}', {})
        
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self.get('dataset', {})
        
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.get('evaluation', {})
        
    def get_video_config(self) -> Dict[str, Any]:
        """Get video processing configuration."""
        return self.get('video', {})
        
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self.get('paths', {})
        
    def resolve_path(self, path_key: str, relative_to: Optional[str] = None) -> Path:
        """
        Resolve a path from configuration.
        
        Args:
            path_key: Configuration key for the path
            relative_to: Base path for relative paths
            
        Returns:
            Resolved Path object
        """
        path = self.get(path_key)
        if path is None:
            raise ValueError(f"Path not found in configuration: {path_key}")
            
        path = Path(path)
        
        if not path.is_absolute() and relative_to:
            path = Path(relative_to) / path
            
        return path
        
    def create_directories(self):
        """Create directories specified in paths configuration."""
        paths_config = self.get_paths_config()
        
        for path_name, path_value in paths_config.items():
            path = Path(path_value)
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {path}")
            
    def print_config(self):
        """Print current configuration."""
        print("=== Current Configuration ===")
        self._print_dict(self.config, indent=0)
        
    def _print_dict(self, d: Dict, indent: int = 0):
        """Print dictionary with proper indentation."""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")


# Default configuration instance
default_config = Config()