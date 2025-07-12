#!/usr/bin/env python3
"""
Basic tracking example - track objects in a video file.
"""

import sys
from pathlib import Path

# Add src to path to import yolo_tracker
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from yolo_tracker import YOLOTracker


def main():
    """Run basic tracking example."""
    
    # Initialize tracker
    print("Initializing YOLO tracker...")
    tracker = YOLOTracker(model_path='yolov8n.pt')
    
    # Create a test video
    test_video = "test_video.mp4"
    print(f"Creating test video: {test_video}")
    
    if not tracker.create_test_video(test_video, width=640, height=480, duration=5, fps=30):
        print("Failed to create test video")
        return 1
        
    # Track objects in the video
    output_video = "tracked_output.mp4"
    print(f"Tracking objects, saving to: {output_video}")
    
    if tracker.track_video(test_video, output_video):
        print("Tracking completed successfully!")
        print(f"Output video saved as: {output_video}")
        return 0
    else:
        print("Tracking failed")
        return 1


if __name__ == "__main__":
    exit(main())