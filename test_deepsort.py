#!/usr/bin/env python3
"""
Test script for deepsort_from_gt.py

This script creates sample test data and runs the DeepSort tracking script
to verify functionality.
"""

import os
import cv2
import numpy as np
import tempfile
import shutil
from pathlib import Path


def create_test_data():
    """Create test data for DeepSort tracking."""
    
    # Create temporary directory
    test_dir = tempfile.mkdtemp()
    print(f"Creating test data in: {test_dir}")
    
    # Create directory structure
    seq_dir = Path(test_dir) / "sequences" / "TEST-001"
    det_dir = Path(test_dir) / "detections"
    seq_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test images
    for i in range(1, 11):  # 10 frames
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add frame number
        cv2.putText(img, f'Frame {i}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add some visual elements
        cv2.rectangle(img, (100, 100), (150, 150), (0, 255, 0), 2)
        cv2.rectangle(img, (200, 200), (250, 250), (255, 0, 0), 2)
        
        # Save image
        cv2.imwrite(str(seq_dir / f"{i:06d}.jpg"), img)
    
    # Create ground truth data
    gt_data = []
    for frame in range(1, 11):
        # Object 1: moving from left to right
        x1 = 100 + (frame - 1) * 10
        y1 = 100
        gt_data.append(f"{frame},1,{x1},{y1},50,50,0.9,1,1")
        
        # Object 2: moving from top to bottom
        x2 = 200
        y2 = 200 + (frame - 1) * 5
        gt_data.append(f"{frame},2,{x2},{y2},40,40,0.8,1,1")
    
    # Save ground truth file
    gt_file = det_dir / "TEST-001.txt"
    with open(gt_file, 'w') as f:
        f.write('\n'.join(gt_data))
    
    print(f"Created {len(gt_data)} ground truth annotations")
    print(f"Created 10 test images")
    
    return test_dir


def run_test():
    """Run the test."""
    print("=" * 60)
    print("Testing DeepSort from Ground Truth Script")
    print("=" * 60)
    
    # Create test data
    test_dir = create_test_data()
    
    try:
        # Import the script
        import sys
        sys.path.append('.')
        
        # Run the script
        output_file = Path(test_dir) / "test_output.mp4"
        
        print(f"\nRunning DeepSort tracking...")
        print(f"Data directory: {test_dir}")
        print(f"Output video: {output_file}")
        
        # Test command
        cmd = [
            "python", "deepsort_from_gt.py",
            "--sequence", "TEST-001",
            "--data_dir", str(test_dir),
            "--output", str(output_file)
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        # Run the command
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"\nReturn code: {result.returncode}")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Check if output file was created
        if output_file.exists():
            print(f"\n✅ SUCCESS: Output video created: {output_file}")
            print(f"Video size: {output_file.stat().st_size} bytes")
        else:
            print(f"\n❌ FAILED: Output video not created")
        
    finally:
        # Clean up
        print(f"\nCleaning up test data...")
        shutil.rmtree(test_dir)
        print("Test complete!")


if __name__ == "__main__":
    run_test()