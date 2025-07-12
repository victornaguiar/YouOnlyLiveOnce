"""
Multi-object tracking using YOLO detection and BotSort tracker.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import os
import collections
from typing import Dict, List, Tuple, Optional

from ultralytics import YOLO
from boxmot import BotSort


class YOLOTracker:
    """YOLO-based object tracker using BotSort."""
    
    def __init__(self, model_path: str = 'yolov8n.pt', tracker_config: Optional[Dict] = None):
        """
        Initialize the YOLO tracker.
        
        Args:
            model_path: Path to YOLO model weights
            tracker_config: Configuration for BotSort tracker
        """
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config or {}
        self.tracker = None
        
    def track_video(self, video_path: str, output_path: str, show_progress: bool = True) -> bool:
        """
        Track objects in a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save tracked video
            show_progress: Whether to show tracking progress
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Error opening video file: {video_path}")
                
            # Get video properties
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            frame_count = 0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                    
                # Run YOLO tracking
                results = self.model.track(frame, persist=True, tracker="botsort.yaml")
                
                # Visualize results
                annotated_frame = results[0].plot()
                
                # Write frame
                out.write(annotated_frame)
                
                frame_count += 1
                if show_progress and frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
                    
            cap.release()
            out.release()
            
            if show_progress:
                print(f"Tracking completed. Output saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error during tracking: {e}")
            return False
    
    def track_with_custom_detections(self, detections_file: str, sequence_path: str, 
                                   output_file: str, img_size: Tuple[int, int] = (640, 480)) -> bool:
        """
        Track objects using pre-computed detections.
        
        Args:
            detections_file: Path to detections file (format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z)
            sequence_path: Path to sequence image directory
            output_file: Path to save tracking results
            img_size: Image size (width, height)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load detections
            detections_by_frame = self._load_detections(detections_file)
            
            # Initialize BotSort tracker
            tracker = BotSort(
                model_weights=Path('osnet_x0_25_msmt17.pt'),
                device='cuda' if torch.cuda.is_available() else 'cpu',
                fp16=False
            )
            
            # Process each frame
            results = []
            for frame_num in sorted(detections_by_frame.keys()):
                frame_detections = detections_by_frame[frame_num]
                
                if frame_detections:
                    # Convert to tracker format
                    dets = np.array([[det[1], det[2], det[1] + det[3], det[2] + det[4], det[5]] 
                                   for det in frame_detections])
                    
                    # Update tracker
                    tracks = tracker.update(dets, np.zeros(img_size))
                    
                    # Save results
                    for track in tracks:
                        track_id = int(track[4])
                        x1, y1, x2, y2 = track[:4]
                        w, h = x2 - x1, y2 - y1
                        results.append([frame_num, track_id, x1, y1, w, h, 1, -1, -1, -1])
                        
            # Save results to file
            self._save_tracking_results(results, output_file)
            return True
            
        except Exception as e:
            print(f"Error during custom tracking: {e}")
            return False
    
    def _load_detections(self, detection_file_path: str) -> Dict[int, List]:
        """Load detections from file."""
        detections_by_frame = collections.defaultdict(list)
        
        with open(detection_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_num = int(parts[0])
                obj_id = int(parts[1])  # Not used for detection
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                confidence = float(parts[6])
                
                detections_by_frame[frame_num].append([
                    frame_num, bb_left, bb_top, bb_width, bb_height, confidence
                ])
                
        return detections_by_frame
    
    def _save_tracking_results(self, results: List, output_file: str):
        """Save tracking results to file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            for result in results:
                line = ','.join(map(str, result))
                f.write(line + '\n')
    
    def create_test_video(self, output_path: str, width: int = 640, height: int = 480, 
                         duration: int = 5, fps: int = 30) -> bool:
        """
        Create a test video with moving objects for testing purposes.
        
        Args:
            output_path: Path to save test video
            width: Video width
            height: Video height
            duration: Duration in seconds
            fps: Frames per second
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            total_frames = fps * duration
            square_size = 50
            
            for i in range(total_frames):
                # Create frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add moving square
                x_pos = int((i / total_frames) * (width - square_size))
                y_pos = int(height / 2 - square_size / 2)
                
                cv2.rectangle(frame, (x_pos, y_pos), 
                            (x_pos + square_size, y_pos + square_size), 
                            (0, 255, 0), -1)
                
                # Add frame number
                cv2.putText(frame, f"Frame {i+1}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(frame)
                
            out.release()
            print(f"Test video created: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating test video: {e}")
            return False