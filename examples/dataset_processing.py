#!/usr/bin/env python3
"""
Dataset processing example - process a custom dataset for tracking evaluation.
"""

import sys
from pathlib import Path

# Add src to path to import yolo_tracker
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from yolo_tracker import YOLOTracker, DatasetManager


def main():
    """Run dataset processing example."""
    
    # Setup dataset directory structure
    dataset_root = Path("example_dataset")
    print(f"Setting up dataset in: {dataset_root}")
    
    dataset_manager = DatasetManager(str(dataset_root))
    dataset_manager.setup_directories()
    
    # Validate dataset structure
    print("Validating dataset structure...")
    validation = dataset_manager.validate_dataset_structure()
    
    for check, passed in validation.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    
    # Generate sequence info files (if sequences exist)
    sequences = dataset_manager.list_sequences()
    if sequences:
        print(f"Found sequences: {sequences}")
        print("Generating sequence info files...")
        dataset_manager.generate_sequence_info_files()
    else:
        print("No sequences found. Dataset setup completed.")
        print(f"You can add sequence directories to: {dataset_manager.sequences_dir}")
        print(f"You can add detection files to: {dataset_manager.detections_dir}")
    
    # Example of tracking with custom detections (if detection files exist)
    detection_files = list(dataset_manager.detections_dir.glob("*.txt"))
    if detection_files:
        print("Found detection files, running custom tracking example...")
        
        tracker = YOLOTracker()
        detection_file = detection_files[0]
        sequence_name = detection_file.stem
        sequence_dir = dataset_manager.sequences_dir / sequence_name
        output_file = dataset_manager.results_dir / f"{sequence_name}_tracked.txt"
        
        if sequence_dir.exists():
            success = tracker.track_with_custom_detections(
                str(detection_file),
                str(sequence_dir),
                str(output_file)
            )
            
            if success:
                print(f"Custom tracking completed: {output_file}")
            else:
                print("Custom tracking failed")
        else:
            print(f"Sequence directory not found: {sequence_dir}")
    
    print("Dataset processing example completed.")
    return 0


if __name__ == "__main__":
    exit(main())