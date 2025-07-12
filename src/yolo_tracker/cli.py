"""
Command-line interface for YOLO tracker.
"""

import argparse
import sys
from pathlib import Path

from .tracker import YOLOTracker
from .evaluator import TrackEvaluator
from .dataset import DatasetManager
from .config import Config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="YOLO Multi-Object Tracker")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Track command
    track_parser = subparsers.add_parser('track', help='Track objects in video')
    track_parser.add_argument('input', help='Input video file')
    track_parser.add_argument('output', help='Output video file')
    track_parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    track_parser.add_argument('--config', help='Configuration file')
    track_parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    
    # Track with detections command
    track_det_parser = subparsers.add_parser('track-detections', 
                                           help='Track using pre-computed detections')
    track_det_parser.add_argument('detections', help='Detections file')
    track_det_parser.add_argument('sequence', help='Sequence directory')
    track_det_parser.add_argument('output', help='Output tracking file')
    track_det_parser.add_argument('--config', help='Configuration file')
    track_det_parser.add_argument('--width', type=int, default=640, help='Image width')
    track_det_parser.add_argument('--height', type=int, default=480, help='Image height')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate tracking results')
    eval_parser.add_argument('gt_folder', help='Ground truth folder')
    eval_parser.add_argument('tracker_folder', help='Tracker results folder')
    eval_parser.add_argument('tracker_name', help='Name of tracker to evaluate')
    eval_parser.add_argument('--metrics', nargs='+', default=['HOTA', 'CLEAR', 'Identity'],
                           help='Metrics to compute')
    eval_parser.add_argument('--benchmark', default='MOT17', help='Benchmark name')
    eval_parser.add_argument('--split', default='train', help='Dataset split')
    eval_parser.add_argument('--config', help='Configuration file')
    
    # Dataset management commands
    dataset_parser = subparsers.add_parser('dataset', help='Dataset management')
    dataset_subparsers = dataset_parser.add_subparsers(dest='dataset_command')
    
    # Generate sequence info
    seqinfo_parser = dataset_subparsers.add_parser('generate-seqinfo',
                                                  help='Generate seqinfo.ini files')
    seqinfo_parser.add_argument('dataset_root', help='Dataset root directory')
    
    # Validate dataset
    validate_parser = dataset_subparsers.add_parser('validate', help='Validate dataset structure')
    validate_parser.add_argument('dataset_root', help='Dataset root directory')
    
    # Copy data locally
    copy_parser = dataset_subparsers.add_parser('copy-local', help='Copy data locally')
    copy_parser.add_argument('source_sequences', help='Source sequences directory')
    copy_parser.add_argument('source_detections', help='Source detections directory')
    copy_parser.add_argument('source_results', help='Source results directory')
    copy_parser.add_argument('local_root', help='Local root directory')
    
    # Demo/test commands
    demo_parser = subparsers.add_parser('demo', help='Run demo')
    demo_parser.add_argument('--output', default='demo_output', help='Output directory')
    demo_parser.add_argument('--config', help='Configuration file')
    
    # Create test video
    test_video_parser = subparsers.add_parser('create-test-video', help='Create test video')
    test_video_parser.add_argument('output', help='Output video file')
    test_video_parser.add_argument('--width', type=int, default=640, help='Video width')
    test_video_parser.add_argument('--height', type=int, default=480, help='Video height')
    test_video_parser.add_argument('--duration', type=int, default=5, help='Duration in seconds')
    test_video_parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    
    # Config commands
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_command')
    
    show_config_parser = config_subparsers.add_parser('show', help='Show current configuration')
    show_config_parser.add_argument('--config', help='Configuration file')
    
    create_config_parser = config_subparsers.add_parser('create', help='Create default configuration')
    create_config_parser.add_argument('output', help='Output configuration file')
    create_config_parser.add_argument('--format', choices=['json', 'yaml'], default='yaml')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # Load configuration
    config = Config(args.config if hasattr(args, 'config') and args.config else None)
    
    try:
        if args.command == 'track':
            return cmd_track(args, config)
        elif args.command == 'track-detections':
            return cmd_track_detections(args, config)
        elif args.command == 'evaluate':
            return cmd_evaluate(args, config)
        elif args.command == 'dataset':
            return cmd_dataset(args, config)
        elif args.command == 'demo':
            return cmd_demo(args, config)
        elif args.command == 'create-test-video':
            return cmd_create_test_video(args, config)
        elif args.command == 'config':
            return cmd_config(args, config)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_track(args, config):
    """Handle track command."""
    tracker = YOLOTracker(args.model)
    
    if args.device != 'auto':
        config.set('model.device', args.device)
        
    success = tracker.track_video(args.input, args.output)
    return 0 if success else 1


def cmd_track_detections(args, config):
    """Handle track-detections command."""
    tracker = YOLOTracker()
    
    success = tracker.track_with_custom_detections(
        args.detections, args.sequence, args.output, 
        img_size=(args.width, args.height)
    )
    return 0 if success else 1


def cmd_evaluate(args, config):
    """Handle evaluate command."""
    evaluator = TrackEvaluator(args.gt_folder, args.tracker_folder)
    
    if not evaluator.validate_setup():
        print("Evaluation setup validation failed.")
        return 1
        
    results = evaluator.evaluate_tracker(
        args.tracker_name, 
        benchmark=args.benchmark,
        split=args.split,
        metrics=args.metrics
    )
    
    evaluator.print_summary(results)
    return 0 if results['success'] else 1


def cmd_dataset(args, config):
    """Handle dataset commands."""
    if args.dataset_command == 'generate-seqinfo':
        dataset_manager = DatasetManager(args.dataset_root)
        success = dataset_manager.generate_sequence_info_files()
        return 0 if success else 1
        
    elif args.dataset_command == 'validate':
        dataset_manager = DatasetManager(args.dataset_root)
        validation = dataset_manager.validate_dataset_structure()
        
        print("Dataset Validation Results:")
        for check, passed in validation.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
            
        all_passed = all(validation.values())
        return 0 if all_passed else 1
        
    elif args.dataset_command == 'copy-local':
        dataset_manager = DatasetManager('')  # Root not needed for this operation
        success = dataset_manager.copy_data_locally(
            args.source_sequences, args.source_detections,
            args.source_results, args.local_root
        )
        return 0 if success else 1
        
    else:
        print(f"Unknown dataset command: {args.dataset_command}")
        return 1


def cmd_demo(args, config):
    """Handle demo command."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test video
    tracker = YOLOTracker()
    test_video = output_dir / "test_video.mp4"
    
    print("Creating test video...")
    if not tracker.create_test_video(str(test_video)):
        print("Failed to create test video")
        return 1
        
    # Track the test video
    tracked_video = output_dir / "tracked_video.mp4"
    print("Tracking test video...")
    if not tracker.track_video(str(test_video), str(tracked_video)):
        print("Failed to track video")
        return 1
        
    print(f"Demo completed successfully. Output in: {output_dir}")
    return 0


def cmd_create_test_video(args, config):
    """Handle create-test-video command."""
    tracker = YOLOTracker()
    success = tracker.create_test_video(
        args.output, args.width, args.height, args.duration, args.fps
    )
    return 0 if success else 1


def cmd_config(args, config):
    """Handle config commands."""
    if args.config_command == 'show':
        config.print_config()
        return 0
        
    elif args.config_command == 'create':
        if args.format == 'json':
            output_path = args.output if args.output.endswith('.json') else args.output + '.json'
        else:
            output_path = args.output if args.output.endswith(('.yaml', '.yml')) else args.output + '.yaml'
            
        config.save_config(output_path)
        print(f"Default configuration saved to: {output_path}")
        return 0
        
    else:
        print(f"Unknown config command: {args.config_command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())