#!/usr/bin/env python3
"""
DeepSort Tracking from Ground Truth Detections

This script performs object tracking using the DeepSort algorithm on the SoccerNet dataset,
utilizing the provided ground truth bounding boxes as input. This approach allows for the
evaluation and optimization of the DeepSort tracker itself, without the influence of a
separate object detection model.

Usage:
    python deepsort_from_gt.py [--sequence SEQUENCE] [--output OUTPUT]

Requirements:
    - deep-sort-realtime
    - opencv-python
    - numpy
    - pandas
    - tqdm

Author: Generated for SoccerNet DeepSort Evaluation
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("ERROR: deep-sort-realtime not found. Please install it with:")
    print("pip install deep-sort-realtime")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DeepSort tracking from ground truth detections'
    )
    parser.add_argument(
        '--sequence', 
        type=str, 
        default='SNMOT-116',
        help='Sequence name (default: SNMOT-116)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/content/drive/MyDrive/SOCCER_DATA',
        help='Base directory containing the dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video path (default: {sequence}_tracked_output.mp4)'
    )
    parser.add_argument(
        '--max_age',
        type=int,
        default=30,
        help='Maximum age for tracks (default: 30)'
    )
    parser.add_argument(
        '--n_init',
        type=int,
        default=3,
        help='Number of consecutive detections before track is confirmed (default: 3)'
    )
    parser.add_argument(
        '--max_iou_distance',
        type=float,
        default=0.7,
        help='Maximum IOU distance for track matching (default: 0.7)'
    )
    parser.add_argument(
        '--embedder',
        type=str,
        default='mobilenet',
        choices=['mobilenet', 'torchreid', 'clip_RN50', 'clip_RN101', 'clip_ViT-B/32'],
        help='Feature embedder to use (default: mobilenet)'
    )
    
    return parser.parse_args()


def load_ground_truth(gt_file):
    """
    Load ground truth data from MOT format file.
    
    Args:
        gt_file (str): Path to the ground truth file
        
    Returns:
        pd.DataFrame: DataFrame with columns [frame, id, x, y, w, h, confidence, class, visibility]
    """
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    print(f"Loading ground truth from: {gt_file}")
    
    # Read the CSV file
    try:
        gt = pd.read_csv(gt_file, header=None)
        
        # Standard MOT format: frame, id, x, y, w, h, confidence, class, visibility
        base_columns = ['frame', 'id', 'x', 'y', 'w', 'h']
        if len(gt.columns) >= 6:
            gt.columns = base_columns + [f'extra_{i}' for i in range(len(gt.columns) - 6)]
        else:
            raise ValueError(f"Expected at least 6 columns, got {len(gt.columns)}")
        
        # Convert frame numbers to integers
        gt['frame'] = gt['frame'].astype(int)
        
        # Add confidence column if not present
        if 'confidence' not in gt.columns and len(gt.columns) > 6:
            gt.rename(columns={'extra_0': 'confidence'}, inplace=True)
        elif 'confidence' not in gt.columns:
            gt['confidence'] = 1.0
        
        # Ensure confidence is numeric
        gt['confidence'] = pd.to_numeric(gt['confidence'], errors='coerce').fillna(1.0)
        
        print(f"Loaded {len(gt)} detections from {gt['frame'].nunique()} frames")
        print(f"Unique IDs: {gt['id'].nunique()}")
        
        return gt
        
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        raise


def get_image_files(img_dir):
    """
    Get sorted list of image files from directory.
    
    Args:
        img_dir (str): Path to images directory
        
    Returns:
        list: Sorted list of image file paths
    """
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
    
    # Support common image formats
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    img_files = []
    
    for ext in extensions:
        img_files.extend(Path(img_dir).glob(f'*{ext}'))
        img_files.extend(Path(img_dir).glob(f'*{ext.upper()}'))
    
    img_files = sorted(img_files)
    print(f"Found {len(img_files)} image files in {img_dir}")
    
    return img_files


def extract_frame_number(img_path):
    """
    Extract frame number from image filename.
    
    Args:
        img_path (Path): Path to image file
        
    Returns:
        int: Frame number
    """
    # Try to extract number from filename
    filename = img_path.stem
    
    # Look for number in filename
    import re
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])  # Take the last number found
    else:
        # Fallback: use the index in sorted order
        return 1


def draw_tracks(frame, tracks, colors=None):
    """
    Draw tracking results on frame.
    
    Args:
        frame (np.ndarray): Input frame
        tracks (list): List of track objects
        colors (dict): Color mapping for track IDs
        
    Returns:
        np.ndarray: Frame with tracking overlays
    """
    if colors is None:
        colors = {}
    
    # Draw tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        x1, y1, x2, y2 = track.to_ltrb()
        
        # Generate color for this track ID
        if track_id not in colors:
            np.random.seed(track_id)
            colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
        
        color = colors[track_id]
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw track ID
        label = f'ID: {track_id}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                     (int(x1) + label_size[0], int(y1)), color, -1)
        cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame


def main():
    """Main function to run DeepSort tracking."""
    args = parse_args()
    
    # Set up paths
    sequence = args.sequence
    data_dir = Path(args.data_dir)
    
    # Default paths for SoccerNet dataset structure
    img_dir = data_dir / 'deepsort_dataset_train' / 'sequences' / sequence
    gt_file = data_dir / 'deepsort_dataset_train' / 'detections' / f'{sequence}.txt'
    
    # Alternative paths if the above don't exist
    if not img_dir.exists():
        img_dir = data_dir / 'sequences' / sequence
        gt_file = data_dir / 'detections' / f'{sequence}.txt'
    
    # Set output path
    if args.output is None:
        output_path = f'/content/{sequence}_tracked_output.mp4'
    else:
        output_path = args.output
    
    print(f"Processing sequence: {sequence}")
    print(f"Images directory: {img_dir}")
    print(f"Ground truth file: {gt_file}")
    print(f"Output video: {output_path}")
    
    # Load ground truth data
    try:
        gt_data = load_ground_truth(gt_file)
    except Exception as e:
        print(f"Failed to load ground truth: {e}")
        return
    
    # Get image files
    try:
        img_files = get_image_files(img_dir)
        if not img_files:
            print(f"No image files found in {img_dir}")
            return
    except Exception as e:
        print(f"Failed to get image files: {e}")
        return
    
    # Initialize DeepSort tracker
    print("Initializing DeepSort tracker...")
    try:
        tracker = DeepSort(
            max_age=args.max_age,
            n_init=args.n_init,
            max_iou_distance=args.max_iou_distance,
            embedder=args.embedder
        )
        print(f"DeepSort initialized with embedder: {args.embedder}")
    except Exception as e:
        print(f"Failed to initialize DeepSort: {e}")
        print("This might be due to missing PyTorch or other dependencies.")
        print("In Google Colab, make sure to install PyTorch first:")
        print("!pip install torch torchvision")
        return
    
    # Initialize video writer
    video_writer = None
    colors = {}  # Track ID to color mapping
    
    # Process frames
    print("Processing frames...")
    processed_frames = 0
    
    for img_path in tqdm(img_files, desc="Processing frames"):
        try:
            # Extract frame number
            frame_num = extract_frame_number(img_path)
            
            # Load image
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            
            # Initialize video writer with first frame
            if video_writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
                print(f"Video writer initialized: {width}x{height}")
            
            # Get ground truth detections for this frame
            frame_detections = gt_data[gt_data['frame'] == frame_num]
            
            if not frame_detections.empty:
                # Prepare detections for DeepSort
                boxes_xywh = frame_detections[['x', 'y', 'w', 'h']].values
                confidences = frame_detections['confidence'].values
                
                # Convert to the format expected by deep-sort-realtime
                raw_detections = []
                for i, (box, conf) in enumerate(zip(boxes_xywh, confidences)):
                    x, y, w, h = box
                    raw_detections.append(([float(x), float(y), float(w), float(h)], float(conf), None))
                
                # Update tracker
                tracks = tracker.update_tracks(raw_detections, frame=frame)
            else:
                # No detections for this frame
                tracks = tracker.update_tracks([], frame=frame)
            
            # Draw tracks on frame
            frame_with_tracks = draw_tracks(frame, tracks, colors)
            
            # Add frame info
            cv2.putText(frame_with_tracks, f'Frame: {frame_num}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_with_tracks, f'Tracks: {len([t for t in tracks if t.is_confirmed()])}',
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame to video
            video_writer.write(frame_with_tracks)
            processed_frames += 1
            
        except Exception as e:
            print(f"Error processing frame {img_path}: {e}")
            # Print debug info for this frame
            if len(str(e)) < 100:  # Only print debug for specific errors
                print(f"Frame number extracted: {frame_num}")
                print(f"Ground truth for frame: {len(gt_data[gt_data['frame'] == frame_num])} detections")
            continue
    
    # Clean up
    if video_writer is not None:
        video_writer.release()
    
    print(f"Processing complete!")
    print(f"Processed {processed_frames} frames")
    print(f"Output video saved to: {output_path}")
    
    # Print tracking statistics
    if processed_frames > 0:
        print(f"\nTracking Statistics:")
        print(f"- Total frames processed: {processed_frames}")
        print(f"- Unique track IDs seen: {len(colors)}")
        print(f"- Output video: {output_path}")


if __name__ == "__main__":
    main()