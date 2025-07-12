#!/usr/bin/env python3
"""
Configuration example - demonstrates configuration management.
"""

import sys
from pathlib import Path

# Add src to path to import yolo_tracker
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from yolo_tracker.config import Config


def main():
    """Demonstrate configuration management."""
    
    print("=== Configuration Management Example ===\n")
    
    # Create default configuration
    print("1. Default Configuration:")
    config = Config()
    config.print_config()
    
    print("\n" + "="*50)
    
    # Modify configuration
    print("\n2. Modifying Configuration:")
    config.set('model.yolo_model', 'yolov8s.pt')
    config.set('model.confidence_threshold', 0.7)
    config.set('tracker.botsort.track_high_thresh', 0.6)
    
    print("Modified values:")
    print(f"  YOLO model: {config.get('model.yolo_model')}")
    print(f"  Confidence threshold: {config.get('model.confidence_threshold')}")
    print(f"  BotSort high threshold: {config.get('tracker.botsort.track_high_thresh')}")
    
    # Save configuration to file
    config_file = "example_config.yaml"
    print(f"\n3. Saving configuration to: {config_file}")
    config.save_config(config_file)
    
    # Load configuration from file
    print(f"\n4. Loading configuration from: {config_file}")
    loaded_config = Config(config_file)
    
    print("Loaded values:")
    print(f"  YOLO model: {loaded_config.get('model.yolo_model')}")
    print(f"  Confidence threshold: {loaded_config.get('model.confidence_threshold')}")
    print(f"  BotSort high threshold: {loaded_config.get('tracker.botsort.track_high_thresh')}")
    
    # Demonstrate configuration sections
    print("\n5. Configuration Sections:")
    
    model_config = loaded_config.get_model_config()
    print(f"  Model config: {model_config}")
    
    tracker_config = loaded_config.get_tracker_config()
    print(f"  Tracker config: {tracker_config}")
    
    dataset_config = loaded_config.get_dataset_config()
    print(f"  Dataset config: {dataset_config}")
    
    # Demonstrate path resolution
    print("\n6. Path Resolution:")
    paths_config = loaded_config.get_paths_config()
    for path_name, path_value in paths_config.items():
        resolved_path = loaded_config.resolve_path(f'paths.{path_name}')
        print(f"  {path_name}: {path_value} -> {resolved_path}")
    
    # Create directories
    print("\n7. Creating Configured Directories:")
    loaded_config.create_directories()
    
    print("\nConfiguration example completed!")
    return 0


if __name__ == "__main__":
    exit(main())